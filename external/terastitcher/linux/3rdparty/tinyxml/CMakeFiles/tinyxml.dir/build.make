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
include 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/depend.make

# Include the progress variables for this target.
include 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/progress.make

# Include the compile flags for this target's objects.
include 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/flags.make

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlerror.cpp.o: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/flags.make
3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlerror.cpp.o: /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/3rdparty/tinyxml/tinyxmlerror.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nicolas.renier/Programs/TeraStitcher-custom/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlerror.cpp.o"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/3rdparty/tinyxml && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tinyxml.dir/tinyxmlerror.cpp.o -c /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/3rdparty/tinyxml/tinyxmlerror.cpp

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlerror.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tinyxml.dir/tinyxmlerror.cpp.i"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/3rdparty/tinyxml && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/3rdparty/tinyxml/tinyxmlerror.cpp > CMakeFiles/tinyxml.dir/tinyxmlerror.cpp.i

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlerror.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tinyxml.dir/tinyxmlerror.cpp.s"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/3rdparty/tinyxml && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/3rdparty/tinyxml/tinyxmlerror.cpp -o CMakeFiles/tinyxml.dir/tinyxmlerror.cpp.s

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlerror.cpp.o.requires:

.PHONY : 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlerror.cpp.o.requires

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlerror.cpp.o.provides: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlerror.cpp.o.requires
	$(MAKE) -f 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/build.make 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlerror.cpp.o.provides.build
.PHONY : 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlerror.cpp.o.provides

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlerror.cpp.o.provides.build: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlerror.cpp.o


3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinystr.cpp.o: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/flags.make
3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinystr.cpp.o: /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/3rdparty/tinyxml/tinystr.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nicolas.renier/Programs/TeraStitcher-custom/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinystr.cpp.o"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/3rdparty/tinyxml && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tinyxml.dir/tinystr.cpp.o -c /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/3rdparty/tinyxml/tinystr.cpp

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinystr.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tinyxml.dir/tinystr.cpp.i"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/3rdparty/tinyxml && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/3rdparty/tinyxml/tinystr.cpp > CMakeFiles/tinyxml.dir/tinystr.cpp.i

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinystr.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tinyxml.dir/tinystr.cpp.s"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/3rdparty/tinyxml && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/3rdparty/tinyxml/tinystr.cpp -o CMakeFiles/tinyxml.dir/tinystr.cpp.s

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinystr.cpp.o.requires:

.PHONY : 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinystr.cpp.o.requires

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinystr.cpp.o.provides: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinystr.cpp.o.requires
	$(MAKE) -f 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/build.make 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinystr.cpp.o.provides.build
.PHONY : 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinystr.cpp.o.provides

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinystr.cpp.o.provides.build: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinystr.cpp.o


3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlparser.cpp.o: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/flags.make
3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlparser.cpp.o: /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/3rdparty/tinyxml/tinyxmlparser.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nicolas.renier/Programs/TeraStitcher-custom/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlparser.cpp.o"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/3rdparty/tinyxml && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tinyxml.dir/tinyxmlparser.cpp.o -c /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/3rdparty/tinyxml/tinyxmlparser.cpp

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlparser.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tinyxml.dir/tinyxmlparser.cpp.i"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/3rdparty/tinyxml && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/3rdparty/tinyxml/tinyxmlparser.cpp > CMakeFiles/tinyxml.dir/tinyxmlparser.cpp.i

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlparser.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tinyxml.dir/tinyxmlparser.cpp.s"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/3rdparty/tinyxml && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/3rdparty/tinyxml/tinyxmlparser.cpp -o CMakeFiles/tinyxml.dir/tinyxmlparser.cpp.s

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlparser.cpp.o.requires:

.PHONY : 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlparser.cpp.o.requires

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlparser.cpp.o.provides: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlparser.cpp.o.requires
	$(MAKE) -f 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/build.make 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlparser.cpp.o.provides.build
.PHONY : 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlparser.cpp.o.provides

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlparser.cpp.o.provides.build: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlparser.cpp.o


3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxml.cpp.o: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/flags.make
3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxml.cpp.o: /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/3rdparty/tinyxml/tinyxml.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nicolas.renier/Programs/TeraStitcher-custom/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxml.cpp.o"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/3rdparty/tinyxml && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tinyxml.dir/tinyxml.cpp.o -c /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/3rdparty/tinyxml/tinyxml.cpp

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxml.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tinyxml.dir/tinyxml.cpp.i"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/3rdparty/tinyxml && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/3rdparty/tinyxml/tinyxml.cpp > CMakeFiles/tinyxml.dir/tinyxml.cpp.i

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxml.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tinyxml.dir/tinyxml.cpp.s"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/3rdparty/tinyxml && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/3rdparty/tinyxml/tinyxml.cpp -o CMakeFiles/tinyxml.dir/tinyxml.cpp.s

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxml.cpp.o.requires:

.PHONY : 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxml.cpp.o.requires

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxml.cpp.o.provides: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxml.cpp.o.requires
	$(MAKE) -f 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/build.make 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxml.cpp.o.provides.build
.PHONY : 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxml.cpp.o.provides

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxml.cpp.o.provides.build: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxml.cpp.o


# Object files for target tinyxml
tinyxml_OBJECTS = \
"CMakeFiles/tinyxml.dir/tinyxmlerror.cpp.o" \
"CMakeFiles/tinyxml.dir/tinystr.cpp.o" \
"CMakeFiles/tinyxml.dir/tinyxmlparser.cpp.o" \
"CMakeFiles/tinyxml.dir/tinyxml.cpp.o"

# External object files for target tinyxml
tinyxml_EXTERNAL_OBJECTS =

3rdparty/tinyxml/libtinyxml.a: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlerror.cpp.o
3rdparty/tinyxml/libtinyxml.a: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinystr.cpp.o
3rdparty/tinyxml/libtinyxml.a: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlparser.cpp.o
3rdparty/tinyxml/libtinyxml.a: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxml.cpp.o
3rdparty/tinyxml/libtinyxml.a: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/build.make
3rdparty/tinyxml/libtinyxml.a: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nicolas.renier/Programs/TeraStitcher-custom/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX static library libtinyxml.a"
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/3rdparty/tinyxml && $(CMAKE_COMMAND) -P CMakeFiles/tinyxml.dir/cmake_clean_target.cmake
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/3rdparty/tinyxml && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tinyxml.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
3rdparty/tinyxml/CMakeFiles/tinyxml.dir/build: 3rdparty/tinyxml/libtinyxml.a

.PHONY : 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/build

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/requires: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlerror.cpp.o.requires
3rdparty/tinyxml/CMakeFiles/tinyxml.dir/requires: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinystr.cpp.o.requires
3rdparty/tinyxml/CMakeFiles/tinyxml.dir/requires: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxmlparser.cpp.o.requires
3rdparty/tinyxml/CMakeFiles/tinyxml.dir/requires: 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/tinyxml.cpp.o.requires

.PHONY : 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/requires

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/clean:
	cd /home/nicolas.renier/Programs/TeraStitcher-custom/3rdparty/tinyxml && $(CMAKE_COMMAND) -P CMakeFiles/tinyxml.dir/cmake_clean.cmake
.PHONY : 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/clean

3rdparty/tinyxml/CMakeFiles/tinyxml.dir/depend:
	cd /home/nicolas.renier/Programs/TeraStitcher-custom && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src /home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/3rdparty/tinyxml /home/nicolas.renier/Programs/TeraStitcher-custom /home/nicolas.renier/Programs/TeraStitcher-custom/3rdparty/tinyxml /home/nicolas.renier/Programs/TeraStitcher-custom/3rdparty/tinyxml/CMakeFiles/tinyxml.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : 3rdparty/tinyxml/CMakeFiles/tinyxml.dir/depend
