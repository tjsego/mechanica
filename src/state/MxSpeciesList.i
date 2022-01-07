%{
    #include "state/MxSpeciesList.h"

%}

%template(vectorSpecies) std::vector<MxSpecies*>;

// MxSpecies instances currently get garbage collected in python, which deallocates the pointer inserted here
%ignore MxSpeciesList::insert(MxSpecies*);

%include "MxSpeciesList.h"

%extend MxSpeciesList {
    %pythoncode %{
        def __str__(self) -> str:
            return self.str()

        def __len__(self) -> int:
            return self.size()

        def __getattr__(self, item: str):
            if item == 'this':
                return object.__getattr__(self, item)

            result = self[item]
            if result is not None:
                return result
            raise AttributeError

        def __setattr__(self, item: str, value) -> None:
            if item == 'this':
                return object.__setattr__(self, item, value)

            if self.item(item) is not None:
                if not isinstance(value, MxSpecies):
                    raise TypeError("Not a species")
                self.insert(value)
                return
            return object.__setattr__(self, item, value)

        def __getitem__(self, item) -> MxSpecies:
            if isinstance(item, str):
                item = self.index_of(item)

            if item < len(self):
                return self.item(item)
            return None

        def __setitem__(self, item, value) -> None:
            if self.item(item) is not None:
                if not isinstance(value, MxSpecies):
                    raise TypeError("Not a species")
                self.insert(value)

        def __reduce__(self):
            return MxSpeciesList.fromString, (self.toString(),)
    %}
}

%pythoncode %{
    SpeciesList = MxSpeciesList
%}
