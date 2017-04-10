# Behavioral-Cloning Project3

## NVDIA Model 

| Layer(type)        | Output Shape    | Param#  |
| ------------------ |:---------------:| -------:|
| Lambda             | None,75,320,3   |    0    |
| Conv2d_1(Conv2D)   | None,36,158,24  |   1824  |
| Conv2d_2(Conv2D)   | None,16,77,36   |  21636  |
| Conv2d_3(Conv2D)   | None,6,37,48    |  43248  |
| Conv2d_4(Conv2D)   | None,4,35,64    |  27712  |
| Conv2d_5(Conv2D)   | None,2,33,64    |  36928  |
| Dropout_1(Dropout) | None,2,33,64    |    0    |
| Flatten_1(Flatten) | None,4224       |    0    |
