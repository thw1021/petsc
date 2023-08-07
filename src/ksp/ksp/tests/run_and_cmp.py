#!/usr/bin/env python3
"""
# Created: Sun Jul 17 08:25:36 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import os
import shutil
import pathlib
import subprocess
import math
import copy

def subprocess_run(*args, **kwargs):
  kwargs.setdefault('capture_output',     True)
  kwargs.setdefault('universal_newlines', True)
  args = list(map(str, args))
  ret  = subprocess.run(args, **kwargs)
  if ret.returncode != 0:
    mess = '\n'.join([
      '',
      '=' * 80,
      f'Subproccess error, return code {ret.returncode}',
      'Command:',
      ' '.join(args),
      '',
      'stdout:',
      ret.stdout,
      'stderr:',
      ret.stderr,
      '=' * 80
    ])
    raise RuntimeError(mess)
  if ret.stderr:
    print('')
    print(ret.stderr)
  return ret.stdout

def run_exec(exec_path, ns_args, *args):
  name = ns_args.ksp_type
  if ns_args.use_cpu:
    name += '_cpu'
  if exec_path.stem.endswith('main'):
    name += '_main'

  if not ns_args.show:
    print('-' * 50)
  print(f'{name}: ', end='', flush=True)
  if name.endswith('async'):
    if ns_args.a_args is not None:
      args = list(args) + copy.deepcopy(ns_args.a_args)
  elif ns_args.b_args is not None:
    args = list(args) + copy.deepcopy(ns_args.b_args)

  arg_exec = [ns_args.mpiexec, '-n', ns_args.mpiexec_n, exec_path]
  if ns_args.profile:
    assert ns_args.device_type == 'cuda'
    arg_exec = [
      'nsys', 'profile', '--force-overwrite=true', '--cudabacktrace=all',
      '--trace=cuda,osrt,cublas-verbose,cusparse-verbose,nvtx',
      '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '-o', name
    ] + arg_exec

  arg_list = [
    '-log_view', '-n_it', ns_args.num_its, '-n_warmup', ns_args.num_warmup, '-n', ns_args.grid_dim,
    '-ksp_type', ns_args.ksp_type, '-n_solve', ns_args.num_solve
  ]
  if ns_args.use_cpu:
    arg_list.extend(['-vec_type', 'standard', '-mat_type', 'aij'])
  elif ns_args.device_type == 'cuda':
    arg_list.extend(['-vec_type', 'cuda', '-mat_type', 'aijcusparse'])
  else:
    assert ns_args.device_type == 'hip'
    arg_list.extend(['-vec_type', 'hip', '-mat_type', 'aijhipsparse'])
  if ns_args.conv_max_its:
    arg_list.extend(['-ksp_converged_maxits', '-ksp_max_it', int(ns_args.conv_max_its)])
  if name.endswith('async'):
    arg_list.extend(['-root_device_context_stream_type', 'default_blocking'])
  arg_list.extend(list(args))

  full_arg_list = arg_exec + arg_list
  ret           = subprocess_run(*full_arg_list).splitlines()

  rest_lines       = []
  log_lines        = []
  log_output       = []
  full_log_capture = 0
  log_capture      = 0
  nnz              = 0
  dofs             = 0
  for line in ret:
    if 'Norm of error' in line:
      answer = line
    elif 'KSP type' in line and 'total time' in line:
      timings = line
    elif 'Number of nonzeros' in line:
      nnz = int(line.split(':')[1].strip())
    elif 'DoFs:' in line:
      dofs = int(line.split(':')[1].strip())
    else:
      if full_log_capture or line.startswith('*' * 160):
        full_log_capture = 1
        log_lines.append(line)
        if line.startswith('Using libraries:'):
          full_log_capture = 0
      else:
        rest_lines.append(line)
    if line.lstrip().startswith('--- Event Stage') and 'Timing' in line:
      log_capture += 1
    elif log_capture == 1:
      line = line.strip()
      if line.startswith('---'):
        log_capture += 1
      elif line:
        log_output.append(line)
  if ns_args.show:
    if len(rest_lines):
      print('\n'.join(rest_lines))
    if ns_agrs.log_view and len(log_lines):
      print('\n'.join(log_lines))
  e_norm, iterations = answer.split(',')

  def extract_time(name):
    return float(timings.partition(name)[2].split(',')[0].strip('. s'))

  return {
    'name'     : name,
    'e_norm'   : float(e_norm.split()[-1].replace('nan.', 'nan')),
    'iter'     : int(iterations.split()[-1]),
    't_total'  : extract_time('total time'),
    't_avg'    : extract_time('avg'),
    't_min'    : extract_time('min'),
    'stdout'   : ret,
    'log_view' : log_output,
    'options'  : full_arg_list,
    'nnz'      : nnz,
    'dofs'     : dofs,
    'nsolve'   : ns_args.num_solve
  }

def post_process(ns_args, data_list):
  def extract(data_dict):
    return {l.split()[0] : l.split()[1:] for l in data_dict['log_view'] if not l.startswith('DCtx')}

  def extract_xfer_stats(log_entries):
    total_cpu2gpu = [0, 0]
    total_gpu2cpu = [0, 0]
    for event, event_data in log_entries.items():
      total_cpu2gpu[0] += int(event_data[-5])   # count
      total_cpu2gpu[1] += float(event_data[-4]) # size
      total_gpu2cpu[0] += int(event_data[-3])   # count
      total_gpu2cpu[1] += float(event_data[-2]) # size
    xfer_pad = max(map(len, map(str, [total_cpu2gpu[0], total_gpu2cpu[0]])))
    size_pad = max(map(len, map(str, [total_cpu2gpu[1], total_gpu2cpu[1]])))

    def printer(base_name, item):
      amount, size = item
      nits         = ns_args.num_its
      print(
        f'{base_name} transfers: total {amount:<{xfer_pad}} ({size:<{size_pad}} MB), avg {amount // nits} ({size / nits:<{size_pad}} MB)'
      )

    printer('cpu2gpu', total_cpu2gpu)
    printer('gpu2cpu', total_gpu2cpu)
    return total_cpu2gpu, total_gpu2cpu

  data_list.sort(key=lambda x: x['t_total'])
  max_width             = max(len(x['name']) for x in data_list) + 1
  dlist_0               = data_list[0]
  fastest_log           = extract(dlist_0)
  dlist_0['log_events'] = fastest_log
  if len(fastest_log.keys()):
    max_event_len = max(map(len, fastest_log.keys()))

    print('<---', dlist_0['name'], '--->')
    h2d, d2h       = extract_xfer_stats(fastest_log)
    dlist_0['h2d'] = h2d
    dlist_0['d2h'] = d2h
    for rank, entry in enumerate(data_list[1:]):
      log_entries = extract(entry)
      print('<---', entry['name'], '--->')
      h2d, d2h = extract_xfer_stats(log_entries)
      data_list[rank + 1]['h2d']        = h2d
      data_list[rank + 1]['d2h']        = d2h
      data_list[rank + 1]['log_events'] = log_entries
      if not ns_args.log_view:
        continue

      event_list = [
        (event, float(fastest_log[event][2]), float(event_data[2]))
        for event, event_data in log_entries.items()
      ]

      event_list.sort(key=lambda x: abs(x[1] - x[2]) / min(x[1], x[2]))

      max_call_len = max(len(str(log_entries[name][0])) for name, _, _ in event_list)
      out_list     = []
      for name, ftime, stime in event_list:
        if ftime == stime:
          cmp_op = '='
          descr  = 'equal'
        else:
          pc_slower = int((abs(stime - ftime) / min(ftime, stime)) * 100)
          if ftime < stime:
            cmp_op = '<'
            descr  = 'slower *'
          else:
            cmp_op = '>'
            descr  = 'faster'

        out_list.append(f'{name:<{max_event_len}}: {ftime:<10} ({fastest_log[name][0]:<{max_call_len}} calls) {cmp_op} {stime:<10} ({log_entries[name][0]:<{max_call_len}} calls) -> {pc_slower:>4}% {descr}')

      olist = out_list[0]
      try:
        spacer = olist.index('<')
      except ValueError:
        spacer = olist.index('>')
      else:
        spacer = min(spacer, olist.index('>'))
      header = '<' + ('-' * (max_event_len - 1)) + ' ' + dlist_0['name']
      header += (' ' * (spacer - len(header))) + '| ' + entry['name'] + ' '
      header += ('-' * (len(olist) - len(header) - 1)) + '>'
      print(header, '\n'.join(out_list), sep='\n')

  entry_name = 't_min'
  print('<--- timings --->')
  print('Ranked by:', entry_name)
  for rank, entry in enumerate(sorted(data_list, key=lambda x: x[entry_name])):
    time = entry[entry_name]
    if rank == 0:
      fastest    = time
      out_phrase = 'fastest'
    else:
      out_phrase = f'{int(((time - fastest) / fastest) * 100)}% slower'
    print(f'{rank + 1} {entry["name"] + ":":<{max_width}} {time}s ({out_phrase})')
  return data_list

def run_exp(ns_args, *args):
  os.environ['PETSC_DIR']  = str(ns_args.petsc_dir)
  os.environ['PETSC_ARCH'] = str(ns_args.petsc_arch)

  print(f'using os.environ["PETSC_ARCH"] = {os.environ["PETSC_ARCH"]}')
  print(f'using os.environ["PETSC_DIR"]  = {os.environ["PETSC_DIR"]}')

  print('building PETSc... ', end='', flush=True)
  subprocess_run('make', '-C', ns_args.petsc_dir, 'libs', '-j', '10')
  print('done')

  print(f'building {ns_args.exec_base.stem}... ', end='', flush=True)
  subprocess_run('make', ns_args.exec_base.stem)
  print('done')

  async_exec = ns_args.exec_base.with_stem(ns_args.exec_base.stem + '_async')
  main_exec  = ns_args.exec_base.with_stem(ns_args.exec_base.stem + '_main')

  print(f'copying {ns_args.exec_base.stem} to {async_exec.stem}... ', end='', flush=True)
  shutil.copy2(ns_args.exec_base, async_exec)
  print('done')

  print('=' * 50)
  print(f'{ns_args.grid_dim = }')
  print(f'{ns_args.num_its = }')
  print(f'{ns_args.num_solve = }')
  print(f'{ns_args.num_warmup = }')

  work_list = [(async_exec, ns_args.ksp_type + 'async', False), (async_exec, ns_args.ksp_type, False)]
  if main_exec.exists():
    work_list.append((main_exec, ns_args.ksp_type, False))
  if ns_args.use_cpu:
    work_list.append((async_exec, ns_args.ksp_type, True))

  data_list = []
  for idx, (exec_name, ksp_type, use_cpu) in enumerate(work_list):
    ns_args_copy          = copy.deepcopy(ns_args)
    ns_args_copy.ksp_type = ksp_type
    ns_args_copy.use_cpu  = use_cpu
    latest                = run_exec(exec_name, ns_args_copy, *args)
    print(f'{latest["t_total"]}s')
    print('options:', ' '.join(map(str, latest['options'])))
    if ns_args_copy.a_args is not None or ns_args_copy.b_args is not None:
      if len(data_list):
        last = data_list[-1]
        for key in ('e_norm', 'iter'):
          if isinstance(last[key], float):
            eq = math.isclose(last[key], latest[key], rel_tol=ns_args_copy.rtol)
          else:
            eq = last[key] == latest[key]
          assert eq, f'{last["name"]}[{key}] {last[key]} != {latest["name"]}[{key}] {latest[key]}'
    data_list.append(latest)

  return post_process(ns_args, data_list)

def run_exp_timeit(ns_args, *args, **kwargs):
  import sys
  from time import time as time_time
  import pickle

  t_max           = 25
  ns_args.num_its = 200
  ns_args.dim     = list(sorted(set(map(int, ns_args.dim))))

  ns_args_cpy = copy.deepcopy(ns_args)
  kwargs_cpy  = copy.deepcopy(kwargs)
  args_cpy    = copy.deepcopy(list(args))
  banner      = ' '.join(('xxx', 80 * '=', 'xxx'))
  ret_dict    = {
    'exec_options' : list((vars(ns_args_cpy) | kwargs_cpy).items()) + args_cpy,
    'argv'         : copy.deepcopy(sys.argv),
    'device_type'  : ns_args_cpy.device_type
  }
  dim_dict    = {}
  for dim in ns_args_cpy.dim:
    if dim == 2:
      max_rows = 1473
      step     = 64
    elif dim == 3:
      max_rows = 129
      step     = 8
    else:
      raise NotImplementedError(f'Need a matrix size incrementer for {dim = }')

    # for some reason the 8x8x1 case produces nan. I don't know why
    begin_offset = step if ns_args_cpy.ksp_type == 'tfqmr' else 0
    rng          = range(8 + begin_offset, max_rows, step)
    stencil_dict = {}
    for fd_stencil in [0, 1]:
      ns_args   = copy.deepcopy(ns_args_cpy)
      kwargs    = copy.deepcopy(kwargs_cpy)
      args      = copy.deepcopy(args_cpy) + ['-fd_stencil', fd_stencil, '-dim', dim]
      grid_dict = {}
      for grid_dim in rng:
        ns_args.grid_dim = grid_dim

        before = time_time()
        ret    = run_exp(ns_args, *args, **kwargs)
        after  = time_time()

        t_ratio = t_max / (after - before)
        if t_ratio < 1:
          ns_args.num_its = max(int(ns_args.num_its * t_ratio), 10)

        grid_dict[grid_dim] = ret
      stencil_dict[fd_stencil] = grid_dict
    dim_dict[dim] = stencil_dict
  ret_dict['data'] = dim_dict
  output_file = f'{ns_args.exec_base.stem}_data_{ns_args.ksp_type}.pkl'
  with open(output_file, 'wb') as fd:
    pickle.dump(ret_dict, fd)
  print('wrote to', output_file)
  return

def main(ns_args, *args, **kwargs):
  ns_args.petsc_dir = pathlib.Path(os.environ.get('PETSC_DIR', '~/petsc')).resolve()
  ns_args.profile   = bool(ns_args.profile)
  if ns_args.profile:
    ns_args.num_its = 1
  ns_args.exec_base = ns_args.exec_base.resolve()
  for arg_list_name in ['a_args', 'b_args']:
    alist = getattr(ns_args, arg_list_name, None)
    if alist:
      setattr(ns_args, arg_list_name, alist.split())
  arch_dir = ns_args.petsc_dir/ns_args.petsc_arch
  assert arch_dir.exists() and arch_dir.is_dir(), f'PETSC_ARCH {ns_args.petsc_arch} does not exist, set the right one from options!'

  if ns_args.device_type == 'determine':
    have_cuda   = 0
    have_hip    = 0
    petscconf_h = arch_dir/'include'/'petscconf.h'
    assert petscconf_h.exists()
    for line in petscconf_h.read_text().splitlines():
      if 'PETSC_HAVE_CUDA' in line:
        have_cuda = 1
      elif 'PETSC_HAVE_HIP' in line:
        have_hip = 1
    if have_cuda and have_hip:
      raise RuntimeError(f'Arch {ns_args.petsc_arch} has both CUDA and HIP. You must explicitly pick which backend to use from options')
    elif have_cuda:
      ns_args.device_type = 'cuda'
    elif have_hip:
      ns_args.device_type = 'hip'
    else:
      raise RuntimeError(f'Arch {ns_args.petsc_arch} has neither CUDA nor HIP?')

  if ns_args.mpiexec == 'determine':
    import re

    mpiexec        = None
    mpiexec_re     = re.compile('MPIEXEC\s+=\s+')
    petscvariables = arch_dir/'lib'/'petsc'/'conf'/'petscvariables'
    assert petscvariables.exists()
    for line in petscvariables.read_text().splitlines():
      if mpiexec_re.search(line):
        mpiexec = line.split('=')[1].strip()
        break
    assert mpiexec, 'Could not determine mpiexec. You must explicitly set it from options'
    ns_args.mpiexec = mpiexec
  ns_args.mpiexec = pathlib.Path(ns_args.mpiexec).resolve()

  if ns_args.timeit:
    run_exp_timeit(ns_args, *args, **kwargs)
  else:
    run_exp(ns_args, *args, **kwargs)
  return

if __name__ == '__main__':
  import argparse

  def required_length(nmin, nmax):
    class RequiredLength(argparse.Action):
      def __call__(self, parser, args, values, option_string=None):
        dest = self.dest
        if not nmin <= len(values) <= nmax:
          raise argparse.ArgumentTypeError(
            f'argument "{dest}" requires between {nmin} and {nmax} arguments'
          )
        setattr(args, dest, values)
    return RequiredLength

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('exec_base', type=pathlib.Path, help='Base file name to benchmark')
  parser.add_argument(
    '--device-type', default='determine', choices=('determine', 'cuda', 'hip'),
    help='type of device backend petsc was configured with'
  )
  parser.add_argument('--petsc-arch', default='arch-cuda-opt', help='override PETSC_ARCH')
  parser.add_argument('--num-its', type=int, default=1000, help='number of interations in timing loop')
  parser.add_argument('--num-solve', type=int, default=1, help='number of solves in timing loop')
  parser.add_argument('--num-warmup', type=int, default=1, help='number of iterations in warmup loop')
  parser.add_argument('--grid-dim', type=int, default=10, help='DMDA stencil grid dim')
  parser.add_argument('--ksp-type', default='cg', help='Base KSPType to profile')
  group = parser.add_mutually_exclusive_group()
  group.add_argument('--profile', action='store_true', help='profile using cuda')
  group.add_argument('--log-view', action='store_true', help='parse log_view')
  parser.add_argument('--show', action='store_true', help='verbose stdout')
  parser.add_argument(
    '--conv-max-its', type=int, nargs='?', const=10, default=0, help='Use -ksp_converged_maxits [VALUE]'
  )
  parser.add_argument(
    '--timeit', action='store_true', help='create a timing run of multiple matrix sizes'
  )
  parser.add_argument(
    '--rtol', type=float, default=1e-12, help='relative tolerance for residual norm check'
  )
  parser.add_argument('--a-args', help='arguments for run \'a\'')
  parser.add_argument('--b-args', help='arguments for run \'b\'')
  parser.add_argument('--mpiexec-n', type=int, default=1, help='number of MPI ranks')
  parser.add_argument('--use-cpu', action='store_true', help='do a CPU run on async exec as well')
  parser.add_argument('--dim', nargs='+', action=required_length(1, 3), default=[2, 3], help='dimensions to test')
  parser.add_argument('--mpiexec', default='determine', help='mpiexec command to use')
  args, rest = parser.parse_known_args()

  main(args, *rest)
