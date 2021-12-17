static char help[] = "Test coloring for finite difference Jacobians with DMStag\n\n";

#include <petscdm.h>
#include <petscdmstag.h>
#include <petscsnes.h>

/*
   Note that DMStagGetValuesStencil and DMStagSetValuesStencil are inefficient,
   compared to DMStagVecGetArray() and friends, and only used here for testing
   purposes, as they allow the code for the Jacobian and residual functions to
   be more similar. In the intended application, where users are not writing
   their own Jacobian assembly routines, one should use the faster, array-based
   approach.
*/

/* A "diagonal" objective function which doesn't use coupling. */
PetscErrorCode FormFunction1DNoCoupling(SNES snes, Vec x, Vec f, void *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          start,n,n_extra,N,dof[2];
  Vec               x_local;
  DM                dm;

  PetscFunctionBegin;
  (void) ctx;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&x_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,x,INSERT_VALUES,x_local);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm,&start,NULL,NULL,&n,NULL,NULL,&n_extra,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&N,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetDOF(dm,&dof[0],&dof[1],NULL,NULL);CHKERRQ(ierr);
  for (PetscInt e=start; e<start+n+n_extra; ++e) {
    for (PetscInt c=0; c<dof[0]; ++c) {
      DMStagStencil row;
      PetscScalar   val_x,val_f;

      row.i = e;
      row.loc = DMSTAG_LEFT;
      row.c = c;
      ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
      val_f = (10.0 + c) * val_x * val_x * val_x;  // f_i = (10 +c) * x_i^3
      ierr = DMStagVecSetValuesStencil(dm,f,1,&row,&val_f,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (e < N) {
      for (PetscInt c=0; c<dof[1]; ++c) {
        DMStagStencil row;
        PetscScalar   val_x,val_f;

        row.i = e;
        row.loc = DMSTAG_ELEMENT;
        row.c = c;
        ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
        val_f = (20.0 + c) * val_x * val_x * val_x;  // f_i = (20 + c) * x_i^3
        ierr = DMStagVecSetValuesStencil(dm,f,1,&row,&val_f,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMRestoreLocalVector(dm,&x_local);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian1DNoCoupling(SNES snes,Vec x,Mat Amat,Mat Pmat,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       start,n,n_extra,N,dof[2];
  Vec            x_local;
  DM             dm;

  PetscFunctionBegin;
  (void) ctx;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&x_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,x,INSERT_VALUES,x_local);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm,&start,NULL,NULL,&n,NULL,NULL,&n_extra,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&N,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetDOF(dm,&dof[0],&dof[1],NULL,NULL);CHKERRQ(ierr);
  for (PetscInt e=start; e<start+n+n_extra; ++e) {
    for (PetscInt c=0; c<dof[0]; ++c) {
      DMStagStencil row_vertex;
      PetscScalar   val_x, val_diagonal;

      row_vertex.i = e;
      row_vertex.loc = DMSTAG_LEFT;
      row_vertex.c = c;
      ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row_vertex,&val_x);CHKERRQ(ierr);
      val_diagonal = 3.0 * (10.0 + c) * val_x * val_x;
      ierr = DMStagMatSetValuesStencil(dm,Amat,1,&row_vertex,1,&row_vertex,&val_diagonal,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (e < N) {
      for (PetscInt c=0; c<dof[1]; ++c) {
        DMStagStencil row_element;
        PetscScalar   val_x,val_diagonal;

        row_element.i = e;
        row_element.loc = DMSTAG_ELEMENT;
        row_element.c = c;
        ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row_element,&val_x);CHKERRQ(ierr);
        val_diagonal = 3.0 * (20.0 + c) * val_x * val_x;
        ierr = DMStagMatSetValuesStencil(dm,Amat,1,&row_element,1,&row_element,&val_diagonal,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMRestoreLocalVector(dm,&x_local);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (Amat != Pmat) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented for distinct Amat and Pmat");
  PetscFunctionReturn(0);
}

/* Objective functions which use the DM's stencil width.
   That width can be zero, which might be useful to test, though it's essential
   useless, as coupling is only allowed in terms of the internal representation
   of unknowns, associating them with elements in an assymmetrical way which
   has no physical meaning.
*/
PetscErrorCode FormFunction1D(SNES snes,Vec x,Vec f,void *ctx)
{
  PetscErrorCode    ierr;
  Vec               x_local;
  PetscInt          dim,stencil_width,start,n,n_extra,N,dof[2];
  DMStagStencilType stencil_type;
  DM                dm;

  PetscFunctionBegin;
  (void) ctx;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"DM dimension must be 1");
  ierr = DMStagGetStencilType(dm,&stencil_type);CHKERRQ(ierr);
  if (stencil_type != DMSTAG_STENCIL_STAR && stencil_type != DMSTAG_STENCIL_BOX) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only star and box stencils supported");
  ierr = DMStagGetStencilWidth(dm,&stencil_width);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&x_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,x,INSERT_VALUES,x_local);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm,&start,NULL,NULL,&n,NULL,NULL,&n_extra,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&N,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetDOF(dm,&dof[0],&dof[1],NULL,NULL);CHKERRQ(ierr);

  ierr = VecZeroEntries(f);CHKERRQ(ierr);

  for (PetscInt e=start; e<start+n+n_extra; ++e) {
    DMStagStencil row_vertex,row_element;

    row_vertex.i = e;
    row_vertex.loc = DMSTAG_LEFT;

    row_element.i = e;
    row_element.loc = DMSTAG_ELEMENT;

    for (PetscInt offset=-stencil_width; offset<=stencil_width; ++offset) {
      const PetscInt e_offset = e + offset;

      // vertex --> vertex
      if (e_offset >=0 && e_offset < N+1) { // Does not fully wrap in the periodic case
        DMStagStencil col;
        PetscScalar   val_x,val_f;

        for (PetscInt c_row=0; c_row<dof[0]; ++c_row) {
          row_vertex.c = c_row;
          for (PetscInt c_col=0; c_col<dof[0]; ++c_col) {
            col.c = c_col;
            col.i = e_offset;
            col.loc = DMSTAG_LEFT;
            ierr = DMStagVecGetValuesStencil(dm,x_local,1,&col,&val_x);CHKERRQ(ierr);
            val_f = (10.0 + offset) * val_x * val_x * val_x;
            ierr = DMStagVecSetValuesStencil(dm,f,1,&row_vertex,&val_f,ADD_VALUES);CHKERRQ(ierr);
          }
        }
      }

      // element --> vertex
      if (e_offset >=0 && e_offset < N) { // Does not fully wrap in the periodic case
        DMStagStencil col;
        PetscScalar   val_x,val_f;

        for (PetscInt c_row=0; c_row<dof[0]; ++c_row) {
          row_vertex.c = c_row;
          for (PetscInt c_col=0; c_col<dof[1]; ++c_col) {
            col.c = c_col;
            col.i = e_offset;
            col.loc = DMSTAG_ELEMENT;
            ierr = DMStagVecGetValuesStencil(dm,x_local,1,&col,&val_x);CHKERRQ(ierr);
            val_f = (15.0 + offset) * val_x * val_x * val_x;
            ierr = DMStagVecSetValuesStencil(dm,f,1,&row_vertex,&val_f,ADD_VALUES);CHKERRQ(ierr);
          }
        }
      }

      if (e < N) {
        // vertex --> element
        if (e_offset >=0 && e_offset < N+1) { // Does not fully wrap in the periodic case
          DMStagStencil col;
          PetscScalar   val_x,val_f;

          for (PetscInt c_row=0; c_row<dof[1]; ++c_row) {
            row_element.c = c_row;
            for (PetscInt c_col=0; c_col<dof[0]; ++c_col) {
              col.c = c_col;
              col.i = e_offset;
              col.loc = DMSTAG_LEFT;
              ierr = DMStagVecGetValuesStencil(dm,x_local,1,&col,&val_x);CHKERRQ(ierr);
              val_f = (25.0 + offset) * val_x * val_x * val_x;
              ierr = DMStagVecSetValuesStencil(dm,f,1,&row_element,&val_f,ADD_VALUES);CHKERRQ(ierr);
            }
          }
        }

        // element --> element
        if (e_offset >=0 && e_offset < N) { // Does not fully wrap in the periodic case
          DMStagStencil col;
          PetscScalar   val_x,val_f;

          for (PetscInt c_row=0; c_row<dof[1]; ++c_row) {
            row_element.c = c_row;
            for (PetscInt c_col=0; c_col<dof[1]; ++c_col) {
              col.c = c_col;
              col.i = e_offset;
              col.loc = DMSTAG_ELEMENT;
              ierr = DMStagVecGetValuesStencil(dm,x_local,1,&col,&val_x);CHKERRQ(ierr);
              val_f = (20.0 + offset) * val_x * val_x * val_x;
              ierr = DMStagVecSetValuesStencil(dm,f,1,&row_element,&val_f,ADD_VALUES);CHKERRQ(ierr);
            }
          }
        }

      }
    }
  }
  ierr = DMRestoreLocalVector(dm,&x_local);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian1D(SNES snes, Vec x, Mat Amat, Mat Pmat, void *ctx)
{
  PetscErrorCode ierr;
  Vec            x_local;
  PetscInt       dim,stencil_width,start,n,n_extra,N,dof[2];
  DM             dm;

  PetscFunctionBegin;
  (void) ctx;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"DM dimension must be 1");
  ierr = DMStagGetStencilWidth(dm,&stencil_width);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&x_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,x,INSERT_VALUES,x_local);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm,&start,NULL,NULL,&n,NULL,NULL,&n_extra,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&N,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetDOF(dm,&dof[0],&dof[1],NULL,NULL);CHKERRQ(ierr);

  ierr = MatZeroEntries(Amat);CHKERRQ(ierr);

  for (PetscInt e=start; e<start+n+n_extra; ++e) {
    DMStagStencil row_vertex,row_element;

    row_vertex.i = e;
    row_vertex.loc = DMSTAG_LEFT;

    row_element.i = e;
    row_element.loc = DMSTAG_ELEMENT;

    for (PetscInt offset=-stencil_width; offset<=stencil_width; ++offset) {
      const PetscInt e_offset = e + offset;

      // vertex --> vertex
      if (e_offset >=0 && e_offset < N+1) {
        DMStagStencil col;
        PetscScalar   val_x,val_j;

        for (PetscInt c_row=0; c_row<dof[0]; ++c_row) {
          row_vertex.c = c_row;
          for (PetscInt c_col=0; c_col<dof[0]; ++c_col) {
            col.c = c_col;
            col.i = e_offset;
            col.loc = DMSTAG_LEFT;
            ierr = DMStagVecGetValuesStencil(dm,x_local,1,&col,&val_x);CHKERRQ(ierr);
            val_j = 3.0 * (10.0 + offset) * val_x * val_x; // TODO update all these values to use c_row and c_col
            ierr = DMStagMatSetValuesStencil(dm,Amat,1,&row_vertex,1,&col,&val_j,ADD_VALUES);CHKERRQ(ierr);
          }
        }
      }

      // element --> vertex
      if (e_offset >=0 && e_offset < N) {
        DMStagStencil col;
        PetscScalar   val_x,val_j;

        for (PetscInt c_row=0; c_row<dof[0]; ++c_row) {
          row_vertex.c = c_row;
          for (PetscInt c_col=0; c_col<dof[1]; ++c_col) {
            col.c = c_col;
            col.i = e_offset;
            col.loc = DMSTAG_ELEMENT;
            ierr = DMStagVecGetValuesStencil(dm,x_local,1,&col,&val_x);CHKERRQ(ierr);
            val_j = 3.0 * (15.0 + offset) * val_x * val_x;
            ierr = DMStagMatSetValuesStencil(dm,Amat,1,&row_vertex,1,&col,&val_j,ADD_VALUES);CHKERRQ(ierr);
          }
        }
      }

      if (e < N) {
        // vertex --> element
        if (e_offset >=0 && e_offset < N+1) {
          DMStagStencil col;
          PetscScalar   val_x,val_j;

          for (PetscInt c_row=0; c_row<dof[1]; ++c_row) {
            row_element.c = c_row;
            for (PetscInt c_col=0; c_col<dof[0]; ++c_col) {
              col.c = c_col;
              col.i = e_offset;
              col.loc = DMSTAG_LEFT;
              ierr = DMStagVecGetValuesStencil(dm,x_local,1,&col,&val_x);CHKERRQ(ierr);
              val_j = 3.0 * (25.0 + offset) * val_x * val_x;
              ierr = DMStagMatSetValuesStencil(dm,Amat,1,&row_element,1,&col,&val_j,ADD_VALUES);CHKERRQ(ierr);
            }
          }
        }

        // element --> element
        if (e_offset >=0 && e_offset < N) {
          DMStagStencil col;
          PetscScalar   val_x,val_j;

          for (PetscInt c_row=0; c_row<dof[1]; ++c_row) {
            row_element.c = c_row;
            for (PetscInt c_col=0; c_col<dof[1]; ++c_col) {
              col.c = c_col;
              col.i = e_offset;
              col.loc = DMSTAG_ELEMENT;
              ierr = DMStagVecGetValuesStencil(dm,x_local,1,&col,&val_x);CHKERRQ(ierr);
              val_j = 3.0 * (20.0 + offset) * val_x * val_x;
              ierr = DMStagMatSetValuesStencil(dm,Amat,1,&row_element,1,&col,&val_j,ADD_VALUES);CHKERRQ(ierr);
            }
          }
        }
      }
    }
  }
  ierr = DMRestoreLocalVector(dm,&x_local);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (Amat != Pmat) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented for distinct Amat and Pmat");
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunction2DNoCoupling(SNES snes,Vec x,Vec f,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       start[2],n[2],n_extra[2],N[2],dof[3];
  Vec            x_local;
  DM             dm;

  PetscFunctionBegin;
  (void) ctx;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&x_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,x,INSERT_VALUES,x_local);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm,&start[0],&start[1],NULL,&n[0],&n[1],NULL,&n_extra[0],&n_extra[1],NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&N[0],&N[1],NULL);CHKERRQ(ierr);
  ierr = DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],NULL);CHKERRQ(ierr);
  for (PetscInt ey=start[1]; ey<start[1]+n[1]+n_extra[1]; ++ey) {
    for (PetscInt ex=start[0]; ex<start[0]+n[0]+n_extra[0]; ++ex) {
      for (PetscInt c=0; c<dof[0]; ++c) {
        DMStagStencil row;
        PetscScalar   val_x,val_f;

        row.i = ex;
        row.j = ey;
        row.loc = DMSTAG_DOWN_LEFT;
        row.c = c;
        ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
        val_f = (5.0 + c) * val_x * val_x * val_x;
        ierr = DMStagVecSetValuesStencil(dm,f,1,&row,&val_f,INSERT_VALUES);CHKERRQ(ierr);
      }
      if (ex < N[0]) {
        for (PetscInt c=0; c<dof[1]; ++c) {
          DMStagStencil row;
          PetscScalar   val_x,val_f;

          row.i = ex;
          row.j = ey;
          row.loc = DMSTAG_DOWN;
          row.c = c;
          ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
          val_f = (10.0 + c) * val_x * val_x * val_x;
          ierr = DMStagVecSetValuesStencil(dm,f,1,&row,&val_f,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
      if (ey < N[1]) {
        for (PetscInt c=0; c<dof[1]; ++c) {
          DMStagStencil row;
          PetscScalar   val_x,val_f;

          row.i = ex;
          row.j = ey;
          row.loc = DMSTAG_LEFT;
          row.c = c;
          ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
          val_f = (15.0 + c) * val_x * val_x * val_x;
          ierr = DMStagVecSetValuesStencil(dm,f,1,&row,&val_f,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
      if (ex < N[0] && ey < N[1]) {
        for (PetscInt c=0; c<dof[2]; ++c) {
          DMStagStencil row;
          PetscScalar   val_x,val_f;

          row.i = ex;
          row.j = ey;
          row.loc = DMSTAG_ELEMENT;
          row.c = c;
          ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
          val_f = (20.0 + c) * val_x * val_x * val_x;
          ierr = DMStagVecSetValuesStencil(dm,f,1,&row,&val_f,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = DMRestoreLocalVector(dm,&x_local);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian2DNoCoupling(SNES snes,Vec x,Mat Amat,Mat Pmat,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       start[2],n[2],n_extra[2],N[2],dof[3];
  Vec            x_local;
  DM             dm;

  PetscFunctionBegin;
  (void) ctx;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&x_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,x,INSERT_VALUES,x_local);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm,&start[0],&start[1],NULL,&n[0],&n[1],NULL,&n_extra[0],&n_extra[1],NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&N[0],&N[1],NULL);CHKERRQ(ierr);
  ierr = DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],NULL);CHKERRQ(ierr);
  for (PetscInt ey=start[1]; ey<start[1]+n[1]+n_extra[1]; ++ey) {
    for (PetscInt ex=start[0]; ex<start[0]+n[0]+n_extra[0]; ++ex) {
      for (PetscInt c=0; c<dof[0]; ++c) {
        DMStagStencil row;
        PetscScalar   val_x,val_diagonal;

        row.i = ex;
        row.j = ey;
        row.loc = DMSTAG_DOWN_LEFT;
        row.c = c;
        ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
        val_diagonal = 3.0 * (5.0 + c) * val_x * val_x;
        ierr = DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val_diagonal,INSERT_VALUES);CHKERRQ(ierr);
      }
      if (ex < N[0]) {
        for (PetscInt c=0; c<dof[1]; ++c) {
          DMStagStencil row;
          PetscScalar   val_x,val_diagonal;

          row.i = ex;
          row.j = ey;
          row.loc = DMSTAG_DOWN;
          row.c = c;
          ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
          val_diagonal = 3.0 * (10.0 + c) * val_x * val_x;
          ierr = DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val_diagonal,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
      if (ey < N[1]) {
        for (PetscInt c=0; c<dof[1]; ++c) {
          DMStagStencil row;
          PetscScalar   val_x,val_diagonal;

          row.i = ex;
          row.j = ey;
          row.loc = DMSTAG_LEFT;
          row.c = c;
          ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
          val_diagonal = 3.0 * (15.0 + c) * val_x * val_x;
          ierr = DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val_diagonal,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
      if (ex < N[0] && ey < N[1]) {
        for (PetscInt c=0; c<dof[2]; ++c) {
          DMStagStencil row;
          PetscScalar   val_x,val_diagonal;

          row.i = ex;
          row.j = ey;
          row.loc = DMSTAG_ELEMENT;
          row.c = c;
          ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
          val_diagonal = 3.0 * (20.0 + c) * val_x * val_x;
          ierr = DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val_diagonal,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = DMRestoreLocalVector(dm,&x_local);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (Amat != Pmat) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented for distinct Amat and Pmat");
  PetscFunctionReturn(0);
}

// FIXME 2d main cases

PetscErrorCode FormFunction3DNoCoupling(SNES snes,Vec x,Vec f,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       start[3],n[3],n_extra[3],N[3],dof[4];
  Vec            x_local;
  DM             dm;

  PetscFunctionBegin;
  (void) ctx;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&x_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,x,INSERT_VALUES,x_local);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm,&start[0],&start[1],&start[2],&n[0],&n[1],&n[2],&n_extra[0],&n_extra[1],&n_extra[2]);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&N[0],&N[1],&N[2]);CHKERRQ(ierr);
  ierr = DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],&dof[3]);CHKERRQ(ierr);
  for (PetscInt ez=start[2]; ez<start[2]+n[2]+n_extra[2]; ++ez) {
    for (PetscInt ey=start[1]; ey<start[1]+n[1]+n_extra[1]; ++ey) {
      for (PetscInt ex=start[0]; ex<start[0]+n[0]+n_extra[0]; ++ex) {
        for (PetscInt c=0; c<dof[0]; ++c) {
          DMStagStencil row;
          PetscScalar   val_x,val_f;

          row.i = ex;
          row.j = ey;
          row.k = ez;
          row.loc = DMSTAG_BACK_DOWN_LEFT;
          row.c = c;
          ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
          val_f = (5.0 + c) * val_x * val_x * val_x;
          ierr = DMStagVecSetValuesStencil(dm,f,1,&row,&val_f,INSERT_VALUES);CHKERRQ(ierr);
        }
        if (ez < N[2]) {
          for (PetscInt c=0; c<dof[1]; ++c) {
            DMStagStencil row;
            PetscScalar   val_x,val_f;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_DOWN_LEFT;
            row.c = c;
            ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
            val_f = (50.0 + c) * val_x * val_x * val_x;
            ierr = DMStagVecSetValuesStencil(dm,f,1,&row,&val_f,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
        if (ey < N[1]) {
          for (PetscInt c=0; c<dof[1]; ++c) {
            DMStagStencil row;
            PetscScalar   val_x,val_f;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_BACK_LEFT;
            row.c = c;
            ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
            val_f = (55.0 + c) * val_x * val_x * val_x;
            ierr = DMStagVecSetValuesStencil(dm,f,1,&row,&val_f,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
        if (ex < N[0]) {
          for (PetscInt c=0; c<dof[1]; ++c) {
            DMStagStencil row;
            PetscScalar   val_x,val_f;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_BACK_DOWN;
            row.c = c;
            ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
            val_f = (60.0 + c) * val_x * val_x * val_x;
            ierr = DMStagVecSetValuesStencil(dm,f,1,&row,&val_f,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
        if (ex < N[0] && ez < N[2]) {
          for (PetscInt c=0; c<dof[2]; ++c) {
            DMStagStencil row;
            PetscScalar   val_x,val_f;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_DOWN;
            row.c = c;
            ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
            val_f = (10.0 + c) * val_x * val_x * val_x;
            ierr = DMStagVecSetValuesStencil(dm,f,1,&row,&val_f,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
        if (ey < N[1] && ez < N[2]) {
          for (PetscInt c=0; c<dof[2]; ++c) {
            DMStagStencil row;
            PetscScalar   val_x,val_f;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_LEFT;
            row.c = c;
            ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
            val_f = (15.0 + c) * val_x * val_x * val_x;
            ierr = DMStagVecSetValuesStencil(dm,f,1,&row,&val_f,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
        if (ex < N[0] && ey < N[1]) {
          for (PetscInt c=0; c<dof[2]; ++c) {
            DMStagStencil row;
            PetscScalar   val_x,val_f;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_BACK;
            row.c = c;
            ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
            val_f = (15.0 + c) * val_x * val_x * val_x;
            ierr = DMStagVecSetValuesStencil(dm,f,1,&row,&val_f,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
        if (ex < N[0] && ey < N[1] && ez < N[2]) {
          for (PetscInt c=0; c<dof[3]; ++c) {
            DMStagStencil row;
            PetscScalar   val_x,val_f;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_ELEMENT;
            row.c = c;
            ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
            val_f = (20.0 + c) * val_x * val_x * val_x;
            ierr = DMStagVecSetValuesStencil(dm,f,1,&row,&val_f,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  ierr = DMRestoreLocalVector(dm,&x_local);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian3DNoCoupling(SNES snes,Vec x,Mat Amat,Mat Pmat,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       start[3],n[3],n_extra[3],N[3],dof[4];
  Vec            x_local;
  DM             dm;

  PetscFunctionBegin;
  (void) ctx;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&x_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,x,INSERT_VALUES,x_local);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm,&start[0],&start[1],&start[2],&n[0],&n[1],&n[2],&n_extra[0],&n_extra[1],&n_extra[2]);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&N[0],&N[1],&N[2]);CHKERRQ(ierr);
  ierr = DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],&dof[3]);CHKERRQ(ierr);
  for (PetscInt ez=start[2]; ez<start[2]+n[2]+n_extra[2]; ++ez) {
    for (PetscInt ey=start[1]; ey<start[1]+n[1]+n_extra[1]; ++ey) {
      for (PetscInt ex=start[0]; ex<start[0]+n[0]+n_extra[0]; ++ex) {
        for (PetscInt c=0; c<dof[0]; ++c) {
          DMStagStencil row;
          PetscScalar   val_x,val_diagonal;

          row.i = ex;
          row.j = ey;
          row.k = ez;
          row.loc = DMSTAG_BACK_DOWN_LEFT;
          row.c = c;
          ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
          val_diagonal = 3.0 * (5.0 + c) * val_x * val_x;
          ierr = DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val_diagonal,INSERT_VALUES);CHKERRQ(ierr);
        }
        if (ez < N[2]) {
          for (PetscInt c=0; c<dof[1]; ++c) {
            DMStagStencil row;
            PetscScalar   val_x,val_diagonal;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_DOWN_LEFT;
            row.c = c;
            ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
            val_diagonal = (50.0 + c) * val_x * val_x * val_x;
            ierr = DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val_diagonal,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
        if (ey < N[1]) {
          for (PetscInt c=0; c<dof[1]; ++c) {
            DMStagStencil row;
            PetscScalar   val_x,val_diagonal;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_BACK_LEFT;
            row.c = c;
            ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
            val_diagonal = 3.0 * (55.0 + c) * val_x * val_x;
            ierr = DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val_diagonal,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
        if (ex < N[0]) {
          for (PetscInt c=0; c<dof[1]; ++c) {
            DMStagStencil row;
            PetscScalar   val_x,val_diagonal;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_BACK_DOWN;
            row.c = c;
            ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
            val_diagonal = 3.0 * (60.0 + c) * val_x * val_x;
            ierr = DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val_diagonal,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
        if (ex < N[0] && ez < N[2]) {
          for (PetscInt c=0; c<dof[2]; ++c) {
            DMStagStencil row;
            PetscScalar   val_x,val_diagonal;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_DOWN;
            row.c = c;
            ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
            val_diagonal = 3.0 * (10.0 + c) * val_x * val_x;
            ierr = DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val_diagonal,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
        if (ey < N[1] && ez < N[2]) {
          for (PetscInt c=0; c<dof[2]; ++c) {
            DMStagStencil row;
            PetscScalar   val_x,val_diagonal;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_LEFT;
            row.c = c;
            ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
            val_diagonal = (15.0 + c) * val_x * val_x * val_x;
            ierr = DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val_diagonal,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
        if (ex < N[0] && ey < N[1]) {
          for (PetscInt c=0; c<dof[2]; ++c) {
            DMStagStencil row;
            PetscScalar   val_x,val_diagonal;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_BACK;
            row.c = c;
            ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
            val_diagonal = 3.0 * (15.0 + c) * val_x * val_x;
            ierr = DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val_diagonal,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
        if (ex < N[0] && ey < N[1] && ez < N[2]) {
          for (PetscInt c=0; c<dof[3]; ++c) {
            DMStagStencil row;
            PetscScalar   val_x,val_diagonal;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_ELEMENT;
            row.c = c;
            ierr = DMStagVecGetValuesStencil(dm,x_local,1,&row,&val_x);CHKERRQ(ierr);
            val_diagonal = 3.0 * (20.0 + c) * val_x * val_x;
            ierr = DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val_diagonal,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  ierr = DMRestoreLocalVector(dm,&x_local);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (Amat != Pmat) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented for distinct Amat and Pmat");
  PetscFunctionReturn(0);
}

// FIXME 3d main cases

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  DM             dm;
  PetscInt       dim;
  PetscBool      no_coupling;
  Vec            x,b;
  SNES           snes;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  dim = 3;
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL);CHKERRQ(ierr);
  no_coupling = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-no_coupling",&no_coupling,NULL);CHKERRQ(ierr);

  switch (dim) {
    case 1:
      ierr = DMStagCreate1d(
          PETSC_COMM_WORLD,
          DM_BOUNDARY_NONE,
          4,
          1, 1,
          1,
          DMSTAG_STENCIL_BOX,
          NULL,
          &dm);CHKERRQ(ierr);
      break;
    case 2:
      ierr = DMStagCreate2d(
          PETSC_COMM_WORLD,
          DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
          4, 3,
          PETSC_DECIDE, PETSC_DECIDE,
          1, 1, 1,
          1,
          DMSTAG_STENCIL_BOX,
          NULL, NULL,
          &dm);CHKERRQ(ierr);
      break;
    case 3:
      ierr = DMStagCreate3d(
          PETSC_COMM_WORLD,
          DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
          4, 3, 3,
          PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
          1, 1, 1, 1,
          1,
          DMSTAG_STENCIL_BOX,
          NULL, NULL, NULL,
          &dm);CHKERRQ(ierr);
      break;
    default: SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension %D",dim);
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,dm);CHKERRQ(ierr);
  if (no_coupling) {
    switch (dim) {
    case 1:
      ierr = SNESSetFunction(snes,NULL,FormFunction1DNoCoupling,NULL);CHKERRQ(ierr);
      ierr = SNESSetJacobian(snes,NULL,NULL,FormJacobian1DNoCoupling,NULL);CHKERRQ(ierr);
      break;
    case 2:
      ierr = SNESSetFunction(snes,NULL,FormFunction2DNoCoupling,NULL);CHKERRQ(ierr);
      ierr = SNESSetJacobian(snes,NULL,NULL,FormJacobian2DNoCoupling,NULL);CHKERRQ(ierr);
      break;
    case 3:
      ierr = SNESSetFunction(snes,NULL,FormFunction3DNoCoupling,NULL);CHKERRQ(ierr);
      ierr = SNESSetJacobian(snes,NULL,NULL,FormJacobian3DNoCoupling,NULL);CHKERRQ(ierr);
      break;
    default: SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension %D",dim);
    }
  } else {
    switch (dim) {
      case 1:
        ierr = SNESSetFunction(snes,NULL,FormFunction1D,NULL);CHKERRQ(ierr);
        ierr = SNESSetJacobian(snes,NULL,NULL,FormJacobian1D,NULL);CHKERRQ(ierr);
        break;
      default: SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension %D",dim);
    }
  }
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecSet(x,2.0);CHKERRQ(ierr); // Initial guess
  ierr = VecSet(b,0.0);CHKERRQ(ierr); // RHS
  ierr = SNESSolve(snes,b,x);CHKERRQ(ierr);

  ierr = SNESDestroy(&snes);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

// FIXME 2d main tests
// FIXME 3d main tests

// FIXME 3d no coupling test:
#if 0
   test:
      suffix: 3d_no_coupling
      nsize: {{1 9}separate output}
      args: -dim 3 -no_coupling -stag_stencil_type none -pc_type jacobi -snes_monitor -snes_test_jacobian -stag_dof_0 {{1 2}separate output} -stag_dof_1 {{1 2}separate output} -stag_dof_2 {{1 2}separate output} -stag_dof_3 {{1 2}separate output}

#endif

/*TEST

   test:
      suffix: 1d_no_coupling
      nsize: {{1 2}separate output}
      args: -dim 1 -no_coupling -stag_stencil_type none -pc_type jacobi -snes_monitor -snes_test_jacobian -stag_dof_0 {{1 2}separate output} -stag_dof_1 {{1 2}separate output}
   test:
      suffix: 1d_test_jac
      nsize: {{1 2}separate output}
      args: -dim 1 -stag_stencil_width {{0 1}separate output} -pc_type jacobi -snes_monitor -snes_converged_reason -snes_test_jacobian
   test:
      suffix: 1d_fd_coloring
      nsize: {{1 2}separate output}
      args: -dim 1 -stag_stencil_width {{0 1 2}separate output} -pc_type jacobi -snes_monitor -snes_converged_reason -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type {{natural sl}}
   test:
      suffix: 1d_periodic
      nsize: {{1 2}separate output}
      args: -dim 1 -stag_boundary_type_x periodic -stag_stencil_width {{1 2}separate output} -pc_type jacobi -snes_monitor -snes_converged_reason -snes_test_jacobian
   test:
      suffix: 1d_multidof
      nsize: 2
      args: -dim 1 -stag_stencil_width 2 -stag_dof_0 2 -stag_dof_1 3 -pc_type jacobi -snes_monitor -snes_converged_reason -snes_test_jacobian
   test:
      suffix: 2d_no_coupling
      nsize: {{1 4}separate output}
      args: -dim 2 -no_coupling -stag_stencil_type none -pc_type jacobi -snes_monitor -snes_test_jacobian -stag_dof_0 {{1 2}separate output} -stag_dof_1 {{1 2}separate output} -stag_dof_2 {{1 2}separate output}
TEST*/


