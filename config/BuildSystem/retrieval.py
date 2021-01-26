from __future__ import absolute_import
import logger

import os
try:
  from urllib import urlretrieve
except ImportError:
  from urllib.request import urlretrieve
try:
  import urlparse as urlparse_local # novermin
except ImportError:
  from urllib import parse as urlparse_local
import config.base
import socket

# Fix parsing for nonstandard schemes
urlparse_local.uses_netloc.extend(['bk', 'ssh', 'svn'])

class Retriever(logger.Logger):
  def __init__(self, sourceControl, argDB = None):
    logger.Logger.__init__(self, argDB)
    self.sourceControl = sourceControl
    self.stamp = None
    self.setup()
    self.saveLog()
    return

  def getAuthorizedUrl(self, url):
    '''This returns a tuple of the unauthorized and authorized URLs for the given URL, and a flag indicating which was input'''
    (scheme, location, path, parameters, query, fragment) = urlparse_local.urlparse(url)
    if not location:
      url     = urlparse_local.urlunparse(('', '', path, parameters, query, fragment))
      authUrl = None
      wasAuth = 0
    else:
      index = location.find('@')
      if index >= 0:
        login   = location[0:index]
        authUrl = url
        url     = urlparse_local.urlunparse((scheme, location[index+1:], path, parameters, query, fragment))
        wasAuth = 1
      else:
        login   = location.split('.')[0]
        authUrl = urlparse_local.urlunparse((scheme, login+'@'+location, path, parameters, query, fragment))
        wasAuth = 0
    return (url, authUrl, wasAuth)

  def testAuthorizedUrl(self, authUrl):
    '''Raise an exception if the URL cannot receive an SSH login without a password'''
    if not authUrl:
      raise RuntimeError('URL is empty')
    (scheme, location, path, parameters, query, fragment) = urlparse_local.urlparse(authUrl)
    return self.executeShellCommand('echo "quit" | ssh -oBatchMode=yes '+location, log = self.log)

  @staticmethod
  def isDirectoryGitRepo(directory):
    import os.path as op
    import subprocess as sp

    def process(output): return str(output.decode(encoding='UTF-8',errors='replace')).strip()

    def gitRevParse(opt, directory, fail=True):
      # we don't use Script.runShellCommand() not to introduce circular dependency
      p = sp.Popen(['git', 'rev-parse'] + [opt], cwd=directory, stdout=sp.PIPE, stderr=sp.PIPE)
      (out, err) = p.communicate()
      if p.returncode and fail:
        raise RuntimeError('git rev-parse %s in %s exited with unexpected error %d: %s' % (opt, ret, err))
      out = process(out)
      err = process(err)
      return (out, err, p.returncode)

    if not op.isdir(directory):
      raise RuntimeError(directory + ' is not a directory')
    directory = op.abspath(directory)

    (out, err, ret) = gitRevParse('--is-inside-git-dir', directory, fail=False)
    if ret:
      if 'not a git repository' in err.lower():
        result = False
      else:
        raise RuntimeError('git rev-parse --is-inside-git-dir exited with unexpected error %d: %s' % (ret, err))
    else:
      isInside = (out.lower() == 'true')
      if isInside:
        (out, err, ret) = gitRevParse('--git-dir', directory)
        if out != '.':
          raise RuntimeError('Directory %s is under git directory %s\nSpecify the latter instead.' % (directory, out))
        result = True
      else:
        (out, err, ret) = gitRevParse('--show-prefix', directory)
        result = not out  # out is '' for toplevel directory
    return result

  @staticmethod
  def getDownloadFailureMessage(package, url, filename=None):
    slashFilename = '/'+filename if filename else ''
    return '''\
Unable to download package %s from: %s
* If URL specified manually - perhaps there is a typo?
* If your network is disconnected - please reconnect and rerun ./configure
* Or perhaps you have a firewall blocking the download
* You can run with --with-packages-download-dir=/adirectory and ./configure will instruct you what packages to download manually
* or you can download the above URL manually, to /yourselectedlocation%s
  and use the configure option:
  --download-%s=/yourselectedlocation%s
    ''' % (package.upper(), url, slashFilename, package, slashFilename)

  def retrieve(self, url, root, package):
    raise NotImplementedError('subclasses must override this method')

  @staticmethod
  def getRetrieverByURL(url, sourceControl, argDB = None):
    '''Fetch package from version control repository or tarfile indicated by URL and expand it into root'''

    if url.startswith('dir://'):
      return DirRetriever(sourceControl, argDB)
    elif url.startswith('link://'):
      return LinkRetriever(sourceControl, argDB)
    elif url.startswith('git://'):
      return GitRetriever(sourceControl, argDB)
    elif url.startswith('hg://') or url.startswith('ssh://hg@'):
      return HgRetriever(sourceControl, argDB)
    elif os.path.isdir(url):
      if Retriever.isDirectoryGitRepo(url):
        return GitRetriever(sourceControl, argDB)
      else:
        return DirRetriever(sourceControl, argDB)
    else:
      return TarballRetriever(sourceControl, argDB)
    return

class DirRetriever(Retriever):
  def retrieve(self, url, root, package):
    import shutil
    self.logPrint('Retrieving %s as directory' % url, 3, 'install')
    d = url[6:] if url.startswith('dir://') else url
    if not os.path.isdir(d): raise RuntimeError('URL %s is not a directory' % url)

    if os.path.isdir(os.path.join(root,os.path.basename(d))): shutil.rmtree(os.path.join(root,os.path.basename(d)))
    if os.path.isfile(os.path.join(root,os.path.basename(d))): os.unlink(os.path.join(root,os.path.basename(d)))

    shutil.copytree(d,os.path.join(root,os.path.basename(d)))
    return

class LinkRetriever(Retriever):
  def retrieve(self, url, root, package):
    import shutil
    self.logPrint('Retrieving %s as link' % url, 3, 'install')
    d = url[7:] if url.startswith('link://') else url
    if not os.path.isdir(d): raise RuntimeError('URL %s is not pointing to a directory' % url)

    if os.path.islink(os.path.join(root,os.path.basename(d))): os.unlink(os.path.join(root,os.path.basename(d)))
    #TODO this impossible - error would be raised above
    if os.path.isfile(os.path.join(root,os.path.basename(d))): os.unlink(os.path.join(root,os.path.basename(d)))
    if os.path.isdir(os.path.join(root,os.path.basename(d))): shutil.rmtree(os.path.join(root,os.path.basename(d)))
    os.symlink(os.path.abspath(d),os.path.join(root,os.path.basename(d)))
    return

class GitRetriever(Retriever):
  def retrieve(self, url, root, package):
    self.logPrint('Retrieving %s as git repo' % url, 3, 'install')
    #TODO error should be raised rather than silent return?
    if not hasattr(self.sourceControl, 'git'): return
    import shutil
    d = url[6:] if url.startswith('git://') else url
    if os.path.isdir(d) and not self.isDirectoryGitRepo(d):
      raise RuntimeError('URL %s is a directory but not a git repository' % url)

    newgitrepo = os.path.join(root,'git.'+package)
    if os.path.isdir(newgitrepo): shutil.rmtree(newgitrepo)
    if os.path.isfile(newgitrepo): os.unlink(newgitrepo)

    try:
      config.base.Configure.executeShellCommand('%s clone %s %s' % (self.sourceControl.git, d, newgitrepo), log = self.log)
    except  RuntimeError as e:
      self.logPrint('ERROR: '+str(e))
      err = str(e)
      failureMessage = self.getDownloadFailureMessage(package, url)
      raise RuntimeError('Unable to clone '+package+'\n'+err+failureMessage)
    return

class HgRetriever(Retriever):
  def retrieve(self, url, root, package):
    self.logPrint('Retrieving %s as hg repo' % url, 3, 'install')
    #TODO error should be raised rather than silent return?
    if not hasattr(self.sourceControl, 'hg'): return
    d = url[5:] if url.startswith('hg://') else url

    newgitrepo = os.path.join(root,'hg.'+package)
    if os.path.isdir(newgitrepo): shutil.rmtree(newgitrepo)
    if os.path.isfile(newgitrepo): os.unlink(newgitrepo)
    try:
      config.base.Configure.executeShellCommand('%s clone %s %s' % (self.sourceControl.hg, d, newgitrepo), log = self.log)
    except  RuntimeError as e:
      self.logPrint('ERROR: '+str(e))
      err = str(e)
      failureMessage = self.getDownloadFailureMessage(package, url)
      raise RuntimeError('Unable to clone '+package+'\n'+err+failureMessage)
    return

class TarballRetriever(Retriever):
  def retrieve(self, url, root, package):
    import shutil
    filename = os.path.basename(urlparse_local.urlparse(url)[2])
    localFile = os.path.join(root,'_d_'+filename)
    self.logPrint('Retrieving %s as tarball to %s' % (url,localFile) , 3, 'install')
    ext =  os.path.splitext(localFile)[1]
    if ext not in ['.bz2','.tbz','.gz','.tgz','.zip','.ZIP']:
      raise RuntimeError('Unknown compression type in URL: '+ url)

    if os.path.exists(localFile):
      os.unlink(localFile)

    if os.path.exists(url):
      if not os.path.isfile(url):
        raise RuntimeError('Local path exists but is not a regular file: '+ url)
      # copy local file
      shutil.copyfile(url, localFile)
    else:
      # fetch remote file
      try:
        sav_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(30)
        urlretrieve(url, localFile)
        socket.setdefaulttimeout(sav_timeout)
      except Exception as e:
        socket.setdefaulttimeout(sav_timeout)
        failureMessage = self.getDownloadFailureMessage(package, url, filename)
        raise RuntimeError(failureMessage)

    self.logPrint('Extracting '+localFile)
    if ext in ['.zip','.ZIP']:
      config.base.Configure.executeShellCommand('cd '+root+'; unzip '+localFile, log = self.log)
      output = config.base.Configure.executeShellCommand('cd '+root+'; zipinfo -1 '+localFile+' | head -n 1', log = self.log)
      dirname = os.path.normpath(output[0].strip())
    else:
      failureMessage = '''\
Downloaded package %s from: %s is not a tarball.
[or installed python cannot process compressed files]
* If you are behind a firewall - please fix your proxy and rerun ./configure
  For example at LANL you may need to set the environmental variable http_proxy (or HTTP_PROXY?) to  http://proxyout.lanl.gov
* You can run with --with-packages-download-dir=/adirectory and ./configure will instruct you what packages to download manually
* or you can download the above URL manually, to /yourselectedlocation/%s
  and use the configure option:
  --download-%s=/yourselectedlocation/%s
''' % (package.upper(), url, filename, package, filename)
      import tarfile
      try:
        tf  = tarfile.open(os.path.join(root, localFile))
      except tarfile.ReadError as e:
        raise RuntimeError(str(e)+'\n'+failureMessage)
      if not tf: raise RuntimeError(failureMessage)
      #git puts 'pax_global_header' as the first entry and some tar utils process this as a file
      firstname = tf.getnames()[0]
      if firstname == 'pax_global_header':
        firstmember = tf.getmembers()[1]
      else:
        firstmember = tf.getmembers()[0]
      # some tarfiles list packagename/ but some list packagename/filename in the first entry
      if firstmember.isdir():
        dirname = firstmember.name
      else:
        dirname = os.path.dirname(firstmember.name)
      tf.extractall(root)
      tf.close()

    # fix file permissions for the untared tarballs.
    try:
      # check if 'dirname' is set'
      if dirname:
        config.base.Configure.executeShellCommand('cd '+root+'; chmod -R a+r '+dirname+';find  '+dirname + ' -type d -name "*" -exec chmod a+rx {} \;', log = self.log)
      else:
        self.logPrintBox('WARNING: Could not determine dirname extracted by '+localFile+' to fix file permissions')
    except RuntimeError as e:
      raise RuntimeError('Error changing permissions for '+dirname+' obtained from '+localFile+ ' : '+str(e))
    os.unlink(localFile)
    return
