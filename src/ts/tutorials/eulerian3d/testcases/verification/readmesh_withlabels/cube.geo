//+
SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 1, 1, 1};


//+
Physical Surface("xmin") = {1};
//+
Physical Surface("zmax") = {6};
//+
Physical Surface("xmax") = {2};
//+
Physical Surface("zmin") = {5};
//+
Physical Surface("ymin") = {3};
//+
Physical Surface("ymax") = {4};

//+
Physical Volume("fluid") = {1};
