"""  
 Welcome to the ORF package!
 ---------------------------- 
 In this documentation you will find useful information on how to use the 
 OrderedForest in Python.
 
 To update documentation go to directory `ORFpy` in Anaconda Prompt and run
 ```
 pdoc --html --config show_source_code=False orf --force
 ```



 What objects are documented? 
 ---------------------------- 
 `pdoc` only extracts _public API_ documentation.[^public] 
 All objects (modules, functions, classes, variables) are only 
 considered public if their _identifiers don't begin with an 
 underscore_ ( \\_ ).[^private] 
  
 [^public]: 
     Here, public API refers to the API that is made available 
     to your project end-users, not the public API e.g. of a 
     private class that can be reasonably extended elsewhere 
     by your project developers. 
  
 [^private]: 
     Prefixing private, implementation-specific objects with 
     an underscore is [a common convention]. 
  
 [a common convention]: https://docs.python.org/3/tutorial/classes.html#private-variables 
  
 In addition, if a module defines [`__all__`][__all__], then only 
 the identifiers contained in this list will be considered public. 
 Otherwise, a module's global identifiers are considered public 
 only if they don't begin with an underscore and are defined 
 in this exact module (i.e. not imported from somewhere else). 
"""

from orf.OrderedForest import OrderedForest
from orf._utils import example_data
__all__ = ["OrderedForest", "example_data"]


