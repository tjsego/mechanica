%{
    #include "state/MxStateVector.h"

%}

%include "MxStateVector.h"

%extend MxStateVector {
    %pythoncode %{
        def __str__(self) -> str:
            return self.str()

        def __len__(self) -> int:
            return self.species.size()

        def __getattr__(self, item: str):
            if item == 'this':
                return object.__getattr__(self, item)

            sl = _mechanica.MxStateVector_species_get(self)
            idx = sl.index_of(item)
            if idx >= 0:
                value = _mechanica.MxStateVector_item(self, idx)
                return MxSpeciesValue(value, self, idx)
            raise AttributeError

        def __setattr__(self, item: str, value: float) -> None:
            if item == 'this':
                return object.__setattr__(self, item, value)

            idx = self.species.index_of(item)
            if idx >= 0:
                self.setItem(idx, value)
                return
            return object.__setattr__(self, item, value)

        def __getitem__(self, item: int):
            if item < len(self):
                return self.item(item)
            return None

        def __setitem__(self, item: int, value: float) -> None:
            if item < len(self):
                self.setItem(item, value)

        def __reduce__(self):
            return MxStateVector.fromString, (self.toString(),)
            
    %}
}

%pythoncode %{
    StateVector = MxStateVector
%}
