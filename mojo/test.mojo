from python import Python

fn main() raises:
   Python.add_to_path("local")
   mypython = Python.import_module("mypython")

   values = mypython.gen_random_values(2, 3)
   print(values)

  