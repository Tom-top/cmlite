# This file is configured at cmake time, loaded at cpack time.

# NSIS specific settings
if(CPACK_GENERATOR MATCHES "NSIS")
  set(CPACK_NSIS_MUI_ICON "/home/nicolas.renier/Programs/TeraStitcher-19190f8f56698079b03d6313a69650488b42df77/src/gui/icons\\\\terastitcher.ico")
  set(CPACK_NSIS_HELP_LINK "http:\\\\abria.github.io/TeraStitcher")
  set(CPACK_NSIS_URL_INFO_ABOUT "http:\\\\abria.github.io/TeraStitcher")
  set(CPACK_NSIS_MODIFY_PATH ON)
  set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
  SET(CPACK_NSIS_MUI_FINISHPAGE_RUN "terastitcher-gui.exe")
endif(CPACK_GENERATOR MATCHES "NSIS")

if("${CPACK_GENERATOR}" STREQUAL "PackageMaker")
  set(CPACK_PACKAGE_DEFAULT_LOCATION "/Applications")
endif("${CPACK_GENERATOR}" STREQUAL "PackageMaker")
