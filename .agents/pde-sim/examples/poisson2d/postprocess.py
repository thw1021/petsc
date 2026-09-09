#!/usr/bin/env pvpython
"""
Standalone post-processor for the poisson2d study.
Run with ParaView's pvpython (bundles vtk, numpy, matplotlib).

Reads:
  poisson_solution.vts            (preferred: u, u_exact, error on 129x129 grid)
  poisson_solution_{u,u_exact,error}.bin  (fallback: raw <f8, row-major)
  convergence.csv                 (level,h,dof,L2,Linf,iterations)

Produces PNGs:
  warp_u.png, mesh_render.png, out_of_range_u.png,
  pointwise_error.png, mesh_convergence.png
Solver-convergence plot is SKIPPED: no residual-history data was dumped.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm

# Read/write alongside this script, wherever it lives.
D = os.path.dirname(os.path.abspath(__file__))
N = 129
H = 1.0 / (N - 1)


def load_fields():
    """Prefer the .vts; fall back to raw .bin. Returns dict of (N,N) arrays."""
    vts = os.path.join(D, "poisson_solution.vts")
    fields = {}
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        r = vtk.vtkXMLStructuredGridReader()
        r.SetFileName(vts)
        r.Update()
        g = r.GetOutput()
        dims = [0, 0, 0]
        g.GetDimensions(dims)
        nx, ny = dims[0], dims[1]
        pd = g.GetPointData()
        for i in range(pd.GetNumberOfArrays()):
            nm = pd.GetArrayName(i)
            fields[nm] = vtk_to_numpy(pd.GetArray(i)).reshape(ny, nx)
        print(f"[load] read .vts dims={nx}x{ny} arrays={list(fields)}")
        return fields, (nx, ny)
    except Exception as e:
        print(f"[load] .vts read failed ({e}); falling back to .bin")
        for nm in ("u", "u_exact", "error"):
            a = np.fromfile(os.path.join(D, f"poisson_solution_{nm}.bin"),
                            dtype="<f8")
            fields[nm] = a.reshape(N, N)
        return fields, (N, N)


def make_warp(u):
    x = np.linspace(0, 1, u.shape[1])
    y = np.linspace(0, 1, u.shape[0])
    X, Y = np.meshgrid(x, y)
    warp_scale = 0.3
    Z = u * warp_scale
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, facecolors=cm.viridis(u), rstride=2,
                           cstride=2, linewidth=0, antialiased=True,
                           shade=False)
    m = cm.ScalarMappable(cmap="viridis")
    m.set_array(u)
    cb = fig.colorbar(m, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label("u")
    ax.set_title("Warp by scalar u (warp_scale=0.3)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("0.3 * u")
    ax.view_init(elev=30, azim=-120)
    p = os.path.join(D, "warp_u.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[warp] {p}")
    return p


def make_mesh(u):
    # Render the structured grid itself. 129x129 is dense; draw every 4th line
    # for legibility plus the full boundary, over a faint field backdrop.
    x = np.linspace(0, 1, u.shape[1])
    y = np.linspace(0, 1, u.shape[0])
    fig, ax = plt.subplots(figsize=(7, 7))
    step = 4
    for xi in x[::step]:
        ax.axvline(xi, color="0.35", lw=0.4)
    for yi in y[::step]:
        ax.axhline(yi, color="0.35", lw=0.4)
    # ensure boundary drawn
    for b in (x[0], x[-1]):
        ax.axvline(b, color="k", lw=1.0)
    for b in (y[0], y[-1]):
        ax.axhline(b, color="k", lw=1.0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title(f"Mesh render: {u.shape[1]}x{u.shape[0]} structured grid "
                 f"(every {step}th line shown, h={H:.4g})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    p = os.path.join(D, "mesh_render.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[mesh] {p}")
    return p


def make_out_of_range(u):
    lo, hi = 0.0, 1.0
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under("magenta")   # below nominal minimum
    cmap.set_over("red")        # above nominal maximum
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(u, origin="lower", extent=[0, 1, 0, 1], cmap=cmap,
                   vmin=lo, vmax=hi)
    cb = fig.colorbar(im, ax=ax, extend="both")
    cb.set_label("u")
    below = int((u < lo).sum()); above = int((u > hi).sum())
    ax.set_title(f"Out-of-range highlight, nominal [{lo},{hi}]\n"
                 f"below(magenta)={below}  above(red)={above}  "
                 f"min={u.min():.3e} max={u.max():.6f}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    p = os.path.join(D, "out_of_range_u.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[range] {p} below={below} above={above}")
    return p, below, above


def make_pointwise_error(u, u_exact, error):
    # auxiliary computation: recompute |u_h - u_exact| and cross-check the
    # dumped 'error' field.
    err = np.abs(u - u_exact)
    maxdiff = float(np.max(np.abs(err - error)))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(err, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
    cb = fig.colorbar(im, ax=ax); cb.set_label("|u_h - u_exact|")
    ax.set_title(f"Pointwise error |u_h - u_exact|  (max={err.max():.3e})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    p = os.path.join(D, "pointwise_error.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[error] {p} max={err.max():.6e} vs-dumped-maxdiff={maxdiff:.3e}")
    return p, float(err.max()), maxdiff


def make_convergence():
    csv = os.path.join(D, "convergence.csv")
    data = np.genfromtxt(csv, delimiter=",", names=True)
    h = np.atleast_1d(data["h"])
    L2 = np.atleast_1d(data["L2"])
    Linf = np.atleast_1d(data["Linf"])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.loglog(h, L2, "o-", label="L2 error")
    ax.loglog(h, Linf, "s-", label="Linf error")
    # reference slope-2 line anchored at the finest-h L2 point
    href = np.array([h.min(), h.max()])
    anchor = L2[np.argmin(h)] / (h.min() ** 2)
    ax.loglog(href, anchor * href ** 2, "k--", label="reference slope 2")
    ax.set_xlabel("h"); ax.set_ylabel("error norm")
    ax.set_title("Mesh-refinement convergence (log-log)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    p = os.path.join(D, "mesh_convergence.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    # observed orders
    ordersL2 = np.log(L2[:-1] / L2[1:]) / np.log(h[:-1] / h[1:])
    ordersLi = np.log(Linf[:-1] / Linf[1:]) / np.log(h[:-1] / h[1:])
    print(f"[conv] {p} obsL2={ordersL2} obsLinf={ordersLi}")
    return p


def main():
    f, dims = load_fields()
    u, u_exact, error = f["u"], f["u_exact"], f["error"]
    print(f"[diag] dims={dims} u:[{u.min():.6e},{u.max():.6e}] "
          f"nan={np.isnan(u).any()} inf={np.isinf(u).any()}")
    make_warp(u)
    make_mesh(u)
    make_out_of_range(u)
    make_pointwise_error(u, u_exact, error)
    make_convergence()
    print("[done]")


if __name__ == "__main__":
    main()
