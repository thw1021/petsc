static char help[] = "REPLACE WITH AN ACTUAL EXAMPLE\n\n";

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, NULL,help));
  PetscCall(PetscFinalize());
  return 0;
}
