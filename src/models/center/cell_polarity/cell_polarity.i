%module cell_polarity

%{
    #include "models/center/cell_polarity/cell_polarity.h"
%}

%rename(_CellPolarity) mx::models::center::CellPolarity;

%include cell_polarity.h
