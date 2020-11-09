//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {3, 0, 0, 1.0};
//+
Point(3) = {3, 1, 0, 1.0};
//+
Point(4) = {0, 1, 0, 1.0};
//+
Point(5) = {0, 0, 0, 1.0};
//+
Point(6) = {0, 1, 0, 1.0};
//+
Point(7) = {3, 1, 0, 1.0};
//+
Point(8) = {3, 0, 0, 1.0};
//+
Line(1) = {4, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 4};//+
Point(9) = {1.5, 0.5, 0, 1.0};
//+
Point(10) = {1.5, 0.7, 0, 1.0};
//+
Point(11) = {1.5, 0.3, 0, 1.0};
//+
Circle(5) = {10, 9, 11};
//+
Circle(6) = {11, 9, 10};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Curve Loop(2) = {5, 6};
//+
Plane Surface(1) = {1, 2};
//+
Physical Curve(1) = {1};
//+
Physical Curve(2) = {3};
//+
Physical Curve(3) = {4, 2};
//+
Physical Curve(4) = {5, 6};
//+
Physical Surface(5) = {1};
