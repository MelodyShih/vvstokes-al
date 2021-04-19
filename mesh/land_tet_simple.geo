// Gmsh project created on Wed Feb 10 14:09:21 2021
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {4, 0, 0, 1.0};
//+
Point(3) = {4, 1, 0, 1.0};
//+
Point(4) = {0, 1, 0, 1.0};
//+
Point(5) = {1.8, -0, 0, 1.0};
//+
Point(6) = {1.8, 0.1, 0, 1.0};
//+
Point(7) = {2.2, 0.1, 0, 1.0};
//+
Point(8) = {2.2, -0, 0, 1.0};
//+
Point(9) = {-0, 0.8, 0, 1.0};
//+
Point(10) = {4, 0.8, 0, 1.0};
//+
Line(1) = {4, 9};
//+
Line(2) = {9, 1};
//+
Line(3) = {1, 5};
//+
Line(4) = {5, 6};
//+
Line(5) = {6, 7};
//+
Line(6) = {7, 8};
//+
Line(7) = {8, 2};
//+
Line(8) = {2, 10};
//+
Line(9) = {10, 3};
//+
Line(10) = {3, 4};
//+
Line(16) = {5, 8};
//+
Line(17) = {9, 10};
//+
Curve Loop(1) = {10, 1, 17, 9};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {17, -8, -7, -6, -5, -4, -3, -2};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {5, 6, -16, 4};
//+
Plane Surface(3) = {3};
//+
Extrude {0, 0, 0.25} {
  Surface{1}; Surface{2}; Surface{3}; 
}
Delete{
  Surface{1}; Surface{2}; Surface{3}; 
}
//+
Physical Surface(1) = {5, 15};
//+
Physical Surface(2) = {7, 9};
//+
Physical Surface(3) = {8, 16, 18};
//+
Physical Surface(4) = {1, 2, 3};
//+
Physical Surface(5) = {10, 17, 14};
//+
Physical Volume(6) = {1};
//+
Physical Volume(7) = {2};
//+
Physical Volume(8) = {3};
