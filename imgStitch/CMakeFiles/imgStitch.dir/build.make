# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/chushaobo/cproject/stitchV3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chushaobo/cproject/stitchV3

# Include any dependencies generated for this target.
include imgStitch/CMakeFiles/imgStitch.dir/depend.make

# Include the progress variables for this target.
include imgStitch/CMakeFiles/imgStitch.dir/progress.make

# Include the compile flags for this target's objects.
include imgStitch/CMakeFiles/imgStitch.dir/flags.make

imgStitch/CMakeFiles/imgStitch.dir/imgStitch.cpp.o: imgStitch/CMakeFiles/imgStitch.dir/flags.make
imgStitch/CMakeFiles/imgStitch.dir/imgStitch.cpp.o: imgStitch/imgStitch.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chushaobo/cproject/stitchV3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object imgStitch/CMakeFiles/imgStitch.dir/imgStitch.cpp.o"
	cd /home/chushaobo/cproject/stitchV3/imgStitch && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/imgStitch.dir/imgStitch.cpp.o -c /home/chushaobo/cproject/stitchV3/imgStitch/imgStitch.cpp

imgStitch/CMakeFiles/imgStitch.dir/imgStitch.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/imgStitch.dir/imgStitch.cpp.i"
	cd /home/chushaobo/cproject/stitchV3/imgStitch && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chushaobo/cproject/stitchV3/imgStitch/imgStitch.cpp > CMakeFiles/imgStitch.dir/imgStitch.cpp.i

imgStitch/CMakeFiles/imgStitch.dir/imgStitch.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/imgStitch.dir/imgStitch.cpp.s"
	cd /home/chushaobo/cproject/stitchV3/imgStitch && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chushaobo/cproject/stitchV3/imgStitch/imgStitch.cpp -o CMakeFiles/imgStitch.dir/imgStitch.cpp.s

imgStitch/CMakeFiles/imgStitch.dir/imgStitch.cpp.o.requires:

.PHONY : imgStitch/CMakeFiles/imgStitch.dir/imgStitch.cpp.o.requires

imgStitch/CMakeFiles/imgStitch.dir/imgStitch.cpp.o.provides: imgStitch/CMakeFiles/imgStitch.dir/imgStitch.cpp.o.requires
	$(MAKE) -f imgStitch/CMakeFiles/imgStitch.dir/build.make imgStitch/CMakeFiles/imgStitch.dir/imgStitch.cpp.o.provides.build
.PHONY : imgStitch/CMakeFiles/imgStitch.dir/imgStitch.cpp.o.provides

imgStitch/CMakeFiles/imgStitch.dir/imgStitch.cpp.o.provides.build: imgStitch/CMakeFiles/imgStitch.dir/imgStitch.cpp.o


# Object files for target imgStitch
imgStitch_OBJECTS = \
"CMakeFiles/imgStitch.dir/imgStitch.cpp.o"

# External object files for target imgStitch
imgStitch_EXTERNAL_OBJECTS =

imgStitch/libimgStitch.a: imgStitch/CMakeFiles/imgStitch.dir/imgStitch.cpp.o
imgStitch/libimgStitch.a: imgStitch/CMakeFiles/imgStitch.dir/build.make
imgStitch/libimgStitch.a: imgStitch/CMakeFiles/imgStitch.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chushaobo/cproject/stitchV3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libimgStitch.a"
	cd /home/chushaobo/cproject/stitchV3/imgStitch && $(CMAKE_COMMAND) -P CMakeFiles/imgStitch.dir/cmake_clean_target.cmake
	cd /home/chushaobo/cproject/stitchV3/imgStitch && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/imgStitch.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
imgStitch/CMakeFiles/imgStitch.dir/build: imgStitch/libimgStitch.a

.PHONY : imgStitch/CMakeFiles/imgStitch.dir/build

imgStitch/CMakeFiles/imgStitch.dir/requires: imgStitch/CMakeFiles/imgStitch.dir/imgStitch.cpp.o.requires

.PHONY : imgStitch/CMakeFiles/imgStitch.dir/requires

imgStitch/CMakeFiles/imgStitch.dir/clean:
	cd /home/chushaobo/cproject/stitchV3/imgStitch && $(CMAKE_COMMAND) -P CMakeFiles/imgStitch.dir/cmake_clean.cmake
.PHONY : imgStitch/CMakeFiles/imgStitch.dir/clean

imgStitch/CMakeFiles/imgStitch.dir/depend:
	cd /home/chushaobo/cproject/stitchV3 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chushaobo/cproject/stitchV3 /home/chushaobo/cproject/stitchV3/imgStitch /home/chushaobo/cproject/stitchV3 /home/chushaobo/cproject/stitchV3/imgStitch /home/chushaobo/cproject/stitchV3/imgStitch/CMakeFiles/imgStitch.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : imgStitch/CMakeFiles/imgStitch.dir/depend

