# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nicolas.renier/Programs/TeraStitcher-custom

# Include any dependencies generated for this target.
include stitcher/CMakeFiles/stitcher.dir/depend.make

# Include the progress variables for this target.
include stitcher/CMakeFiles/stitcher.dir/progress.make

# Include the compile flags for this target's objects.
include stitcher/CMakeFiles/stitcher.dir/flags.make

stitcher/CMakeFiles/stitcher.dir/MergeTiles.cpp.o: stitcher/CMakeFiles/stitcher.dir/flags.make
stitcher/CMakeFiles/stitcher.dir/MergeTiles.cpp.o: /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/MergeTiles.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nicolas.renier/Programs/TeraStitcher-custom/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object stitcher/CMakeFiles/stitcher.dir/MergeTiles.cpp.o"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/stitcher.dir/MergeTiles.cpp.o -c /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/MergeTiles.cpp

stitcher/CMakeFiles/stitcher.dir/MergeTiles.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stitcher.dir/MergeTiles.cpp.i"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/MergeTiles.cpp > CMakeFiles/stitcher.dir/MergeTiles.cpp.i

stitcher/CMakeFiles/stitcher.dir/MergeTiles.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stitcher.dir/MergeTiles.cpp.s"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/MergeTiles.cpp -o CMakeFiles/stitcher.dir/MergeTiles.cpp.s

stitcher/CMakeFiles/stitcher.dir/MergeTiles.cpp.o.requires:

.PHONY : stitcher/CMakeFiles/stitcher.dir/MergeTiles.cpp.o.requires

stitcher/CMakeFiles/stitcher.dir/MergeTiles.cpp.o.provides: stitcher/CMakeFiles/stitcher.dir/MergeTiles.cpp.o.requires
	$(MAKE) -f stitcher/CMakeFiles/stitcher.dir/build.make stitcher/CMakeFiles/stitcher.dir/MergeTiles.cpp.o.provides.build
.PHONY : stitcher/CMakeFiles/stitcher.dir/MergeTiles.cpp.o.provides

stitcher/CMakeFiles/stitcher.dir/MergeTiles.cpp.o.provides.build: stitcher/CMakeFiles/stitcher.dir/MergeTiles.cpp.o


stitcher/CMakeFiles/stitcher.dir/StackStitcher.cpp.o: stitcher/CMakeFiles/stitcher.dir/flags.make
stitcher/CMakeFiles/stitcher.dir/StackStitcher.cpp.o: /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/StackStitcher.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nicolas.renier/Programs/TeraStitcher-custom/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object stitcher/CMakeFiles/stitcher.dir/StackStitcher.cpp.o"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/stitcher.dir/StackStitcher.cpp.o -c /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/StackStitcher.cpp

stitcher/CMakeFiles/stitcher.dir/StackStitcher.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stitcher.dir/StackStitcher.cpp.i"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/StackStitcher.cpp > CMakeFiles/stitcher.dir/StackStitcher.cpp.i

stitcher/CMakeFiles/stitcher.dir/StackStitcher.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stitcher.dir/StackStitcher.cpp.s"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/StackStitcher.cpp -o CMakeFiles/stitcher.dir/StackStitcher.cpp.s

stitcher/CMakeFiles/stitcher.dir/StackStitcher.cpp.o.requires:

.PHONY : stitcher/CMakeFiles/stitcher.dir/StackStitcher.cpp.o.requires

stitcher/CMakeFiles/stitcher.dir/StackStitcher.cpp.o.provides: stitcher/CMakeFiles/stitcher.dir/StackStitcher.cpp.o.requires
	$(MAKE) -f stitcher/CMakeFiles/stitcher.dir/build.make stitcher/CMakeFiles/stitcher.dir/StackStitcher.cpp.o.provides.build
.PHONY : stitcher/CMakeFiles/stitcher.dir/StackStitcher.cpp.o.provides

stitcher/CMakeFiles/stitcher.dir/StackStitcher.cpp.o.provides.build: stitcher/CMakeFiles/stitcher.dir/StackStitcher.cpp.o


stitcher/CMakeFiles/stitcher.dir/Displacement.cpp.o: stitcher/CMakeFiles/stitcher.dir/flags.make
stitcher/CMakeFiles/stitcher.dir/Displacement.cpp.o: /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/Displacement.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nicolas.renier/Programs/TeraStitcher-custom/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object stitcher/CMakeFiles/stitcher.dir/Displacement.cpp.o"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/stitcher.dir/Displacement.cpp.o -c /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/Displacement.cpp

stitcher/CMakeFiles/stitcher.dir/Displacement.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stitcher.dir/Displacement.cpp.i"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/Displacement.cpp > CMakeFiles/stitcher.dir/Displacement.cpp.i

stitcher/CMakeFiles/stitcher.dir/Displacement.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stitcher.dir/Displacement.cpp.s"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/Displacement.cpp -o CMakeFiles/stitcher.dir/Displacement.cpp.s

stitcher/CMakeFiles/stitcher.dir/Displacement.cpp.o.requires:

.PHONY : stitcher/CMakeFiles/stitcher.dir/Displacement.cpp.o.requires

stitcher/CMakeFiles/stitcher.dir/Displacement.cpp.o.provides: stitcher/CMakeFiles/stitcher.dir/Displacement.cpp.o.requires
	$(MAKE) -f stitcher/CMakeFiles/stitcher.dir/build.make stitcher/CMakeFiles/stitcher.dir/Displacement.cpp.o.provides.build
.PHONY : stitcher/CMakeFiles/stitcher.dir/Displacement.cpp.o.provides

stitcher/CMakeFiles/stitcher.dir/Displacement.cpp.o.provides.build: stitcher/CMakeFiles/stitcher.dir/Displacement.cpp.o


stitcher/CMakeFiles/stitcher.dir/TPAlgo.cpp.o: stitcher/CMakeFiles/stitcher.dir/flags.make
stitcher/CMakeFiles/stitcher.dir/TPAlgo.cpp.o: /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/TPAlgo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nicolas.renier/Programs/TeraStitcher-custom/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object stitcher/CMakeFiles/stitcher.dir/TPAlgo.cpp.o"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/stitcher.dir/TPAlgo.cpp.o -c /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/TPAlgo.cpp

stitcher/CMakeFiles/stitcher.dir/TPAlgo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stitcher.dir/TPAlgo.cpp.i"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/TPAlgo.cpp > CMakeFiles/stitcher.dir/TPAlgo.cpp.i

stitcher/CMakeFiles/stitcher.dir/TPAlgo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stitcher.dir/TPAlgo.cpp.s"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/TPAlgo.cpp -o CMakeFiles/stitcher.dir/TPAlgo.cpp.s

stitcher/CMakeFiles/stitcher.dir/TPAlgo.cpp.o.requires:

.PHONY : stitcher/CMakeFiles/stitcher.dir/TPAlgo.cpp.o.requires

stitcher/CMakeFiles/stitcher.dir/TPAlgo.cpp.o.provides: stitcher/CMakeFiles/stitcher.dir/TPAlgo.cpp.o.requires
	$(MAKE) -f stitcher/CMakeFiles/stitcher.dir/build.make stitcher/CMakeFiles/stitcher.dir/TPAlgo.cpp.o.provides.build
.PHONY : stitcher/CMakeFiles/stitcher.dir/TPAlgo.cpp.o.provides

stitcher/CMakeFiles/stitcher.dir/TPAlgo.cpp.o.provides.build: stitcher/CMakeFiles/stitcher.dir/TPAlgo.cpp.o


stitcher/CMakeFiles/stitcher.dir/DisplacementMIPNCC.cpp.o: stitcher/CMakeFiles/stitcher.dir/flags.make
stitcher/CMakeFiles/stitcher.dir/DisplacementMIPNCC.cpp.o: /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/DisplacementMIPNCC.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nicolas.renier/Programs/TeraStitcher-custom/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object stitcher/CMakeFiles/stitcher.dir/DisplacementMIPNCC.cpp.o"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/stitcher.dir/DisplacementMIPNCC.cpp.o -c /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/DisplacementMIPNCC.cpp

stitcher/CMakeFiles/stitcher.dir/DisplacementMIPNCC.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stitcher.dir/DisplacementMIPNCC.cpp.i"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/DisplacementMIPNCC.cpp > CMakeFiles/stitcher.dir/DisplacementMIPNCC.cpp.i

stitcher/CMakeFiles/stitcher.dir/DisplacementMIPNCC.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stitcher.dir/DisplacementMIPNCC.cpp.s"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/DisplacementMIPNCC.cpp -o CMakeFiles/stitcher.dir/DisplacementMIPNCC.cpp.s

stitcher/CMakeFiles/stitcher.dir/DisplacementMIPNCC.cpp.o.requires:

.PHONY : stitcher/CMakeFiles/stitcher.dir/DisplacementMIPNCC.cpp.o.requires

stitcher/CMakeFiles/stitcher.dir/DisplacementMIPNCC.cpp.o.provides: stitcher/CMakeFiles/stitcher.dir/DisplacementMIPNCC.cpp.o.requires
	$(MAKE) -f stitcher/CMakeFiles/stitcher.dir/build.make stitcher/CMakeFiles/stitcher.dir/DisplacementMIPNCC.cpp.o.provides.build
.PHONY : stitcher/CMakeFiles/stitcher.dir/DisplacementMIPNCC.cpp.o.provides

stitcher/CMakeFiles/stitcher.dir/DisplacementMIPNCC.cpp.o.provides.build: stitcher/CMakeFiles/stitcher.dir/DisplacementMIPNCC.cpp.o


stitcher/CMakeFiles/stitcher.dir/PDAlgo.cpp.o: stitcher/CMakeFiles/stitcher.dir/flags.make
stitcher/CMakeFiles/stitcher.dir/PDAlgo.cpp.o: /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/PDAlgo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nicolas.renier/Programs/TeraStitcher-custom/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object stitcher/CMakeFiles/stitcher.dir/PDAlgo.cpp.o"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/stitcher.dir/PDAlgo.cpp.o -c /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/PDAlgo.cpp

stitcher/CMakeFiles/stitcher.dir/PDAlgo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stitcher.dir/PDAlgo.cpp.i"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/PDAlgo.cpp > CMakeFiles/stitcher.dir/PDAlgo.cpp.i

stitcher/CMakeFiles/stitcher.dir/PDAlgo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stitcher.dir/PDAlgo.cpp.s"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/PDAlgo.cpp -o CMakeFiles/stitcher.dir/PDAlgo.cpp.s

stitcher/CMakeFiles/stitcher.dir/PDAlgo.cpp.o.requires:

.PHONY : stitcher/CMakeFiles/stitcher.dir/PDAlgo.cpp.o.requires

stitcher/CMakeFiles/stitcher.dir/PDAlgo.cpp.o.provides: stitcher/CMakeFiles/stitcher.dir/PDAlgo.cpp.o.requires
	$(MAKE) -f stitcher/CMakeFiles/stitcher.dir/build.make stitcher/CMakeFiles/stitcher.dir/PDAlgo.cpp.o.provides.build
.PHONY : stitcher/CMakeFiles/stitcher.dir/PDAlgo.cpp.o.provides

stitcher/CMakeFiles/stitcher.dir/PDAlgo.cpp.o.provides.build: stitcher/CMakeFiles/stitcher.dir/PDAlgo.cpp.o


stitcher/CMakeFiles/stitcher.dir/resumer.cpp.o: stitcher/CMakeFiles/stitcher.dir/flags.make
stitcher/CMakeFiles/stitcher.dir/resumer.cpp.o: /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/resumer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nicolas.renier/Programs/TeraStitcher-custom/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object stitcher/CMakeFiles/stitcher.dir/resumer.cpp.o"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/stitcher.dir/resumer.cpp.o -c /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/resumer.cpp

stitcher/CMakeFiles/stitcher.dir/resumer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stitcher.dir/resumer.cpp.i"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/resumer.cpp > CMakeFiles/stitcher.dir/resumer.cpp.i

stitcher/CMakeFiles/stitcher.dir/resumer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stitcher.dir/resumer.cpp.s"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/resumer.cpp -o CMakeFiles/stitcher.dir/resumer.cpp.s

stitcher/CMakeFiles/stitcher.dir/resumer.cpp.o.requires:

.PHONY : stitcher/CMakeFiles/stitcher.dir/resumer.cpp.o.requires

stitcher/CMakeFiles/stitcher.dir/resumer.cpp.o.provides: stitcher/CMakeFiles/stitcher.dir/resumer.cpp.o.requires
	$(MAKE) -f stitcher/CMakeFiles/stitcher.dir/build.make stitcher/CMakeFiles/stitcher.dir/resumer.cpp.o.provides.build
.PHONY : stitcher/CMakeFiles/stitcher.dir/resumer.cpp.o.provides

stitcher/CMakeFiles/stitcher.dir/resumer.cpp.o.provides.build: stitcher/CMakeFiles/stitcher.dir/resumer.cpp.o


stitcher/CMakeFiles/stitcher.dir/TPAlgoMST.cpp.o: stitcher/CMakeFiles/stitcher.dir/flags.make
stitcher/CMakeFiles/stitcher.dir/TPAlgoMST.cpp.o: /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/TPAlgoMST.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nicolas.renier/Programs/TeraStitcher-custom/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object stitcher/CMakeFiles/stitcher.dir/TPAlgoMST.cpp.o"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/stitcher.dir/TPAlgoMST.cpp.o -c /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/TPAlgoMST.cpp

stitcher/CMakeFiles/stitcher.dir/TPAlgoMST.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stitcher.dir/TPAlgoMST.cpp.i"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/TPAlgoMST.cpp > CMakeFiles/stitcher.dir/TPAlgoMST.cpp.i

stitcher/CMakeFiles/stitcher.dir/TPAlgoMST.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stitcher.dir/TPAlgoMST.cpp.s"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/TPAlgoMST.cpp -o CMakeFiles/stitcher.dir/TPAlgoMST.cpp.s

stitcher/CMakeFiles/stitcher.dir/TPAlgoMST.cpp.o.requires:

.PHONY : stitcher/CMakeFiles/stitcher.dir/TPAlgoMST.cpp.o.requires

stitcher/CMakeFiles/stitcher.dir/TPAlgoMST.cpp.o.provides: stitcher/CMakeFiles/stitcher.dir/TPAlgoMST.cpp.o.requires
	$(MAKE) -f stitcher/CMakeFiles/stitcher.dir/build.make stitcher/CMakeFiles/stitcher.dir/TPAlgoMST.cpp.o.provides.build
.PHONY : stitcher/CMakeFiles/stitcher.dir/TPAlgoMST.cpp.o.provides

stitcher/CMakeFiles/stitcher.dir/TPAlgoMST.cpp.o.provides.build: stitcher/CMakeFiles/stitcher.dir/TPAlgoMST.cpp.o


stitcher/CMakeFiles/stitcher.dir/StackRestorer.cpp.o: stitcher/CMakeFiles/stitcher.dir/flags.make
stitcher/CMakeFiles/stitcher.dir/StackRestorer.cpp.o: /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/StackRestorer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nicolas.renier/Programs/TeraStitcher-custom/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object stitcher/CMakeFiles/stitcher.dir/StackRestorer.cpp.o"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/stitcher.dir/StackRestorer.cpp.o -c /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/StackRestorer.cpp

stitcher/CMakeFiles/stitcher.dir/StackRestorer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stitcher.dir/StackRestorer.cpp.i"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/StackRestorer.cpp > CMakeFiles/stitcher.dir/StackRestorer.cpp.i

stitcher/CMakeFiles/stitcher.dir/StackRestorer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stitcher.dir/StackRestorer.cpp.s"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/StackRestorer.cpp -o CMakeFiles/stitcher.dir/StackRestorer.cpp.s

stitcher/CMakeFiles/stitcher.dir/StackRestorer.cpp.o.requires:

.PHONY : stitcher/CMakeFiles/stitcher.dir/StackRestorer.cpp.o.requires

stitcher/CMakeFiles/stitcher.dir/StackRestorer.cpp.o.provides: stitcher/CMakeFiles/stitcher.dir/StackRestorer.cpp.o.requires
	$(MAKE) -f stitcher/CMakeFiles/stitcher.dir/build.make stitcher/CMakeFiles/stitcher.dir/StackRestorer.cpp.o.provides.build
.PHONY : stitcher/CMakeFiles/stitcher.dir/StackRestorer.cpp.o.provides

stitcher/CMakeFiles/stitcher.dir/StackRestorer.cpp.o.provides.build: stitcher/CMakeFiles/stitcher.dir/StackRestorer.cpp.o


stitcher/CMakeFiles/stitcher.dir/PDAlgoMIPNCC.cpp.o: stitcher/CMakeFiles/stitcher.dir/flags.make
stitcher/CMakeFiles/stitcher.dir/PDAlgoMIPNCC.cpp.o: /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/PDAlgoMIPNCC.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nicolas.renier/Programs/TeraStitcher-custom/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object stitcher/CMakeFiles/stitcher.dir/PDAlgoMIPNCC.cpp.o"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/stitcher.dir/PDAlgoMIPNCC.cpp.o -c /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/PDAlgoMIPNCC.cpp

stitcher/CMakeFiles/stitcher.dir/PDAlgoMIPNCC.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stitcher.dir/PDAlgoMIPNCC.cpp.i"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/PDAlgoMIPNCC.cpp > CMakeFiles/stitcher.dir/PDAlgoMIPNCC.cpp.i

stitcher/CMakeFiles/stitcher.dir/PDAlgoMIPNCC.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stitcher.dir/PDAlgoMIPNCC.cpp.s"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher/PDAlgoMIPNCC.cpp -o CMakeFiles/stitcher.dir/PDAlgoMIPNCC.cpp.s

stitcher/CMakeFiles/stitcher.dir/PDAlgoMIPNCC.cpp.o.requires:

.PHONY : stitcher/CMakeFiles/stitcher.dir/PDAlgoMIPNCC.cpp.o.requires

stitcher/CMakeFiles/stitcher.dir/PDAlgoMIPNCC.cpp.o.provides: stitcher/CMakeFiles/stitcher.dir/PDAlgoMIPNCC.cpp.o.requires
	$(MAKE) -f stitcher/CMakeFiles/stitcher.dir/build.make stitcher/CMakeFiles/stitcher.dir/PDAlgoMIPNCC.cpp.o.provides.build
.PHONY : stitcher/CMakeFiles/stitcher.dir/PDAlgoMIPNCC.cpp.o.provides

stitcher/CMakeFiles/stitcher.dir/PDAlgoMIPNCC.cpp.o.provides.build: stitcher/CMakeFiles/stitcher.dir/PDAlgoMIPNCC.cpp.o


# Object files for target stitcher
stitcher_OBJECTS = \
"CMakeFiles/stitcher.dir/MergeTiles.cpp.o" \
"CMakeFiles/stitcher.dir/StackStitcher.cpp.o" \
"CMakeFiles/stitcher.dir/Displacement.cpp.o" \
"CMakeFiles/stitcher.dir/TPAlgo.cpp.o" \
"CMakeFiles/stitcher.dir/DisplacementMIPNCC.cpp.o" \
"CMakeFiles/stitcher.dir/PDAlgo.cpp.o" \
"CMakeFiles/stitcher.dir/resumer.cpp.o" \
"CMakeFiles/stitcher.dir/TPAlgoMST.cpp.o" \
"CMakeFiles/stitcher.dir/StackRestorer.cpp.o" \
"CMakeFiles/stitcher.dir/PDAlgoMIPNCC.cpp.o"

# External object files for target stitcher
stitcher_EXTERNAL_OBJECTS =

stitcher/libstitcher.a: stitcher/CMakeFiles/stitcher.dir/MergeTiles.cpp.o
stitcher/libstitcher.a: stitcher/CMakeFiles/stitcher.dir/StackStitcher.cpp.o
stitcher/libstitcher.a: stitcher/CMakeFiles/stitcher.dir/Displacement.cpp.o
stitcher/libstitcher.a: stitcher/CMakeFiles/stitcher.dir/TPAlgo.cpp.o
stitcher/libstitcher.a: stitcher/CMakeFiles/stitcher.dir/DisplacementMIPNCC.cpp.o
stitcher/libstitcher.a: stitcher/CMakeFiles/stitcher.dir/PDAlgo.cpp.o
stitcher/libstitcher.a: stitcher/CMakeFiles/stitcher.dir/resumer.cpp.o
stitcher/libstitcher.a: stitcher/CMakeFiles/stitcher.dir/TPAlgoMST.cpp.o
stitcher/libstitcher.a: stitcher/CMakeFiles/stitcher.dir/StackRestorer.cpp.o
stitcher/libstitcher.a: stitcher/CMakeFiles/stitcher.dir/PDAlgoMIPNCC.cpp.o
stitcher/libstitcher.a: stitcher/CMakeFiles/stitcher.dir/build.make
stitcher/libstitcher.a: stitcher/CMakeFiles/stitcher.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nicolas.renier/Programs/TeraStitcher-custom/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Linking CXX static library libstitcher.a"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && $(CMAKE_COMMAND) -P CMakeFiles/stitcher.dir/cmake_clean_target.cmake
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stitcher.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
stitcher/CMakeFiles/stitcher.dir/build: stitcher/libstitcher.a

.PHONY : stitcher/CMakeFiles/stitcher.dir/build

stitcher/CMakeFiles/stitcher.dir/requires: stitcher/CMakeFiles/stitcher.dir/MergeTiles.cpp.o.requires
stitcher/CMakeFiles/stitcher.dir/requires: stitcher/CMakeFiles/stitcher.dir/StackStitcher.cpp.o.requires
stitcher/CMakeFiles/stitcher.dir/requires: stitcher/CMakeFiles/stitcher.dir/Displacement.cpp.o.requires
stitcher/CMakeFiles/stitcher.dir/requires: stitcher/CMakeFiles/stitcher.dir/TPAlgo.cpp.o.requires
stitcher/CMakeFiles/stitcher.dir/requires: stitcher/CMakeFiles/stitcher.dir/DisplacementMIPNCC.cpp.o.requires
stitcher/CMakeFiles/stitcher.dir/requires: stitcher/CMakeFiles/stitcher.dir/PDAlgo.cpp.o.requires
stitcher/CMakeFiles/stitcher.dir/requires: stitcher/CMakeFiles/stitcher.dir/resumer.cpp.o.requires
stitcher/CMakeFiles/stitcher.dir/requires: stitcher/CMakeFiles/stitcher.dir/TPAlgoMST.cpp.o.requires
stitcher/CMakeFiles/stitcher.dir/requires: stitcher/CMakeFiles/stitcher.dir/StackRestorer.cpp.o.requires
stitcher/CMakeFiles/stitcher.dir/requires: stitcher/CMakeFiles/stitcher.dir/PDAlgoMIPNCC.cpp.o.requires

.PHONY : stitcher/CMakeFiles/stitcher.dir/requires

stitcher/CMakeFiles/stitcher.dir/clean:
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher && $(CMAKE_COMMAND) -P CMakeFiles/stitcher.dir/cmake_clean.cmake
.PHONY : stitcher/CMakeFiles/stitcher.dir/clean

stitcher/CMakeFiles/stitcher.dir/depend:
	cd /home/nicolas.renier/Programs/TeraStitcher-custom && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/stitcher /home/nicolas.renier/Programs/TeraStitcher-custom /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher /home/nicolas.renier/Programs/TeraStitcher-custom/stitcher/CMakeFiles/stitcher.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : stitcher/CMakeFiles/stitcher.dir/depend
