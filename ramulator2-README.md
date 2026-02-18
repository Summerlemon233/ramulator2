# Ramulator V2.0a
## Introduction
Ramulator 2.0 is a modern, modular, and extensible cycle-accurate DRAM simulator. It is the successor of Ramulator 1.0 [Kim+, CAL'16], achieving both fast simulation speed and ease of extension. The goal of Ramulator 2.0 is to enable rapid and agile implementation and evaluation of design changes in the memory controller and DRAM to meet the increasing research effort in improving the performance, security, and reliability of memory systems. Ramulator 2.0 abstracts and models key components in a DRAM-based memory system and their interactions into shared interfaces and independent implementations, enabling easy modification and extension of the modeled functions of the memory controller and DRAM. 

This Github repository contains the public version of Ramulator 2.0. From time to time, we will synchronize improvements of the code framework, additional functionalities, bug fixes, etc. from our internal version. Ramulator 2.0 is in its early stage and welcomes your contribution as well as new ideas and implementations in the memory system!

Currently, Ramulator 2.0 provides the DRAM models for the following standards:
- DDR3, DDR4, DDR5
- LPDDR5
- GDDR6
- HBM(2), HBM3

Ramulator 2.0 also provides implementations for the following RowHammer mitigation techniques:
- PARA [[Kim+, ISCA'14]](https://people.inf.ethz.ch/omutlu/pub/dram-row-hammer_isca14.pdf)
- TWiCe [[Lee+, ISCA'19]](https://ieeexplore.ieee.org/document/8980327)
- Graphene [[Park+, MICRO'20]](https://microarch.org/micro53/papers/738300a001.pdf)
- Hydra [[Qureshi+, ISCA'22]](https://memlab.ece.gatech.edu/papers/ISCA_2022_1.pdf)
- Randomized Row Swap (RRS) [[Saileshwar+, ASPLOS'22]](https://gururaj-s.github.io/assets/pdf/ASPLOS22_Saileshwar.pdf)
- An "Oracle" Refresh Mitigation [[Kim+, ISCA'20]](https://people.inf.ethz.ch/omutlu/pub/Revisiting-RowHammer_isca20-FINAL-DO-NOT_DISTRIBUTE.pdf)

A quick glance at Ramulator 2.0's other key features:
- Modular and extensible software architecture: Ramulator 2.0 provides an explicit separation of implementations from interfaces. Therefore new ideas can be implemented without intrusive changes.
- Self-registering factory for interface and implementation classes: Ramulator 2.0 automatically constructs the correct class of objects by their names as you specify in the configuration. Do *not* worry about boilerplate code!
- YAML-based configuration file: Ramulator 2.0 is configured via human-readable and machine-friendly configuration files. Sweeping parameters is as easy as editing a Python dictionary!

The initial release of Ramulator 2.0 is described in the following [paper](https://people.inf.ethz.ch/omutlu/pub/Ramulator2_arxiv23.pdf):
> Haocong Luo, Yahya Can Tugrul, F. Nisa Bostancı, Ataberk Olgun, A. Giray Yaglıkcı, and Onur Mutlu,
> "Ramulator 2.0: A Modern, Modular, and Extensible DRAM Simulator,"
> arXiv, 2023.

If you use Ramulator 2.0 in your work, please use the following citation:
```
@misc{luo2023ramulator2,
  title={{Ramulator 2.0: A Modern, Modular, and Extensible DRAM Simulator}}, 
  author={Haocong Luo and Yahya Can Tu\u{g}rul and F. Nisa Bostancı and Ataberk Olgun and A. Giray Ya\u{g}l{\i}k\c{c}{\i} and and Onur Mutlu},
  year={2023},
  archivePrefix={arXiv},
  primaryClass={cs.AR}
}
```

## Using Ramulator 2.0
### Dependencies
Ramulator uses some C++20 features to achieve both high runtime performance and modularity and extensibility. Therefore, a C++20-capable compiler is needed to build Ramulator 2.0. We have tested and verified Ramulator 2.0 with the following compilers:
- `g++-12`
- `clang++-15`

Ramulator 2.0 uses the following external libraries. The build system (CMake) will automatically download and configure these dependencies.
- [argparse](https://github.com/p-ranav/argparse)
- [spdlog](https://github.com/gabime/spdlog)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)
### Getting Started
Clone the repository
```bash
  $ git clone https://github.com/CMU-SAFARI/ramulator2
```
Configure the project and build the executable
```bash
  $ mkdir build
  $ cd build
  $ cmake ..
  $ make -j
  $ cp ./ramulator2 ../ramulator2
  $ cd ..
```
This should produce a `ramulator2` executable that you can execute standalone and a `libramulator.so` dynamic library that can be used as a memory system library by other simulators.
### Running Ramulator 2.0 in Standalone Mode
Ramulator 2.0 comes with two independent simulation frontends: A memory-trace parser and a simplistic out-of-order core model that can accept instruction traces. To start a simulation with these frontends, just run the Ramulator 2.0 executable with the path to the configuration file specified through the `-f` argument
```bash
  $ ./ramulator2 -f ./example_config.yaml
```
To support easy automation of experiments (e.g., evaluate many different traces and sweep parameters), Ramulator 2.0 can accept the configurations as a string dump of the YAML document, which is usually produced by a scripting language that can easily parse and manipulate YAML documents (e.g., `python`). We provide an example `python` snippet to demonstrate an experiment of sweeping the `nRCD` timing constraint:
```python
import os
import yaml  # YAML parsing provided with the PyYAML package

baseline_config_file = "./example_config.yaml"
nRCD_list = [10, 15, 20, 25]

base_config = None
with open(base_config_file, 'r') as f:
  base_config = yaml.safe_load(f)

for nRCD in nRCD_list:
  config["MemorySystem"]["DRAM"]["timing"]["nRCD"] = nRCD
  cmds = ["./ramulator2", str(config)]
  # Run the command with e.g., os.system(), subprocess.run(), ...
```

### General Instructions for Writing Your Own Wrapper of Ramulator 2.0 for Another (including Your Own) Simulator
We describe the key steps and cover the key interfaces involved in using Ramulator 2.0 as a library for your own simulator.
1. Add Ramulator 2.0's key header files to your build system:
- `ramulator2/src/base/base.h`
- `ramulator2/src/base/request.h`
- `ramulator2/src/base/config.h`
- `ramulator2/src/frontend/frontend.h`
- `ramulator2/src/memory_system/memory_system.h`

2. Parse the YAML configuration for Ramulator 2.0 and instantiate the interfaces of the two top-level components, e.g.,
```c++
// MyWrapper.h
std::string config_path;
Ramulator::IFrontEnd* ramulator2_frontend;
Ramulator::IMemorySystem* ramulator2_memorysystem;

// MyWrapper.cpp
YAML::Node config = Ramulator::Config::parse_config_file(config_path, {});
ramulator2_frontend = Ramulator::Factory::create_frontend(config);
ramulator2_memorysystem = Ramulator::Factory::create_memory_system(config);

ramulator2_frontend->connect_memory_system(ramulator2_memorysystem);
ramulator2_memorysystem->connect_frontend(ramulator2_frontend);
```
3. Communicate the necessary memory system information from Ramulator 2.0 to your system (e.g., memory system clock):
```c++
float memory_tCK = ramulator2_memorysystem->get_tCK();
```
4. Send the memory requests from your simulator to Ramulator 2.0, with the correspoding callbacks that should be executed when the request is "completed" by Ramulator 2.0, e.g.,
```c++
if (is_read_request) {
  enqueue_success = ramulator2_frontend->
    receive_external_requests(0, memory_address, context_id, 
    [this](Ramulator::Request& req) {
      // your read request callback 
    });

  if (enqueue_success) {
    // What happens if the memory request is accepted by Ramulator 2.0
  } else {
    // What happens if the memory request is rejected by Ramulator 2.0 (e.g., request queue full)
  }
}
```
5. Find a proper time and place to call the epilogue functions of Ramulator 2.0 when your simulator has finished execution, e.g.,
```c++
void my_simulator_finish() {
  ramulator2_frontend->finalize();
  ramulator2_memorysystem->finalize();
}
```

## Extending Ramulator 2.0
### Directory Structure
Ramulator 2.0 
```
ext                     # External libraries
src                     # Source code of Ramulator 2.0
└ <component1>          # Collection of the source code of all interfaces and implementations related to the component
  └ impl                # Collection of the source code of all implementations of the component
    └ com_impl.cpp      # Source file of a specific implementation
  └ com_interface.h     # Header file that defines an interface
  └ CMakeList.txt       # Component-level CMake configuration
└ ...                    
└ CMakeList.txt         # Top-level CMake configuration of all Ramulator 2.0's source files
CMakeList.txt           # Project-level CMake configuration
```

### Interface and Implementation
To achieve high modularity and extensibility, Ramulator 2.0 models the components in the memory system using two concepts, Interfaces and Implementations:
- An interface is a high-level abstraction of the common high-level functionality of a component as seen by other components in the system. It is an abstract C++ class defined in a `.h` header file, that provides its functionalities through virtual functions.
- An implementation is a concrete realization of an interface that models the actual behavior of the object. It is usually a C++ class that inherits from the interface class it is implementing that provides implementations of the interface's virtual functions.

An example interface class looks like this:
```c++
// example_interface.h
#ifndef     RAMULATOR_EXAMPLE_INTERFACE_H
#define     RAMULATOR_EXAMPLE_INTERFACE_H

// Defines fundamental data structures and types of Ramulator 2.0. Must include for all interfaces.
#include "base/base.h"  

namespace Ramulator {
class ExampleIfce {
  // One-liner macro to register this "ExampleIfce" interface with the name "ExampleInterface" to Ramulator 2.0.
  RAMULATOR_REGISTER_INTERFACE(ExampleIfce, "ExampleInterface", "An example of an interface class.")
  public:
    // Model common behaviors of the interface with virtual functions 
    virtual void foo() = 0;
};
}        // namespace Ramulator

#endif   // RAMULATOR_EXAMPLE_INTERFACE_H
```

An example implementation that implements the above interface looks like this
```c++
// example_impl.cpp
#include <iostream>

// An implementation should always include the header of the interface that it is implementating
#include "example_interface.h"

namespace Ramulator {
// An implementation class should always inherit from *both* the interface it is implementating, and the "Implementation" base class
class ExampleImpl : public ExampleIfce, public Implementation  {
  // One-liner macro to register and bind this "ExampleImpl" implementation with the name "ExampleImplementation" to the "ExampleIfce" interface.
  RAMULATOR_REGISTER_IMPLEMENTATION(ExampleIfce, ExampleImpl, "ExampleImplementation", "An example of an implementation class.")
  public:
    // Implements concrete behavior
    virtual void foo() override {
      std::cout << "Bar!" << std::endl;
    };
};
}      // namespace Ramulator
```

### Adding an Implementation to an Existing Interface
Let us consider an example of adding an implementation "MyImpl" for the interface "MyIfce", defined in `my_ifce.h`, that belongs to a component "my_comp".
1. Create a new `.cpp` file under `my_comp/impl/` that contains the source code for the implementation. Say this file is "my_impl.cpp". The directory structure for `my_comp` will look like this:
```
my_comp                     
  └ impl
    └ my_impl.cpp      
  └ my_interface.h     
  └ CMakeList.txt       
```
2. Provide the concrete class definition for `MyImpl` in `my_impl.cpp`, following the structure explained above. It is important to do the following two things when defining your class:
    * Make sure you inherit from *both* the interface class that you are implementing and the `Implementation` base class
    * Make sure to include the macro `RAMULATOR_REGISTER_IMPLEMENTATION(...)` *inside* your implementation class definition. You should always specify which interface class your class is implementing and the stringified name of your implementation class in the macro.
Assume you have the following registration macro for the interface class
```c++
class MyIfce {
  RAMULATOR_REGISTER_INTERFACE(MyIfce, "MyInterfaceName", "An example of my interface class.")
}
```
and the following for the implementation class
```c++
class MyImpl : public MyIfce, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(MyIfce, MyImpl, "MyImplName", "An example of my implementation class.")
}
```
If everything is correct, Ramulator 2.0 will be able to automatically construct a pointer of type `MyIfce*`, pointing to an object of type `MyImpl`, when it sees the following in the YAML configuration file:
```yaml
MyInterfaceName:
  impl: MyImplName
```

3. Finally, add `impl/my_impl.cpp` to `target_sources` in `CMakeList.txt` so that CMake knows about the newly added source file.

### Adding a New Interface (or New Component)
The process is similar to that of adding a new implementation, but you will need to create the interface and add them to `CMakeList.txt` under the corresponding component directory. If you add a new component (i.e., create a new directory under `src/`) you will need to add this new directory to the `CMakeList.txt` file under `src/`, i.e.,
```cmake
add_subdirectory(my_comp)
```



