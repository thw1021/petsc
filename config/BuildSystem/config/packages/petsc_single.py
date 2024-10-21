import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.liblist         = [['libpetsc_single.a']]
    self.useddirectly    = 1
    self.linkedbypetsc   = 1
    self.builtafterpetsc = 0
    self.functions       = ['__petsc_single_PetscInitialize']
    self.package         = 'petsc-single'
    return

