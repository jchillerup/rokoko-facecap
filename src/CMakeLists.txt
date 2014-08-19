#find_library(MGL mgl PATHS /usr/local/lib)

ADD_EXECUTABLE( rokoko-facestreamer main.cpp findEyeCenter.cpp findEyeCenter.h findEyeCorner.cpp findEyeCorner.h constants.h helpers.cpp helpers.h)
ADD_EXECUTABLE( server debug-server.cpp )
TARGET_LINK_LIBRARIES( rokoko-facestreamer ${OpenCV_LIBS} lo )
TARGET_LINK_LIBRARIES( server ${OpenCV-LIBS} lo )
#target_link_libraries( eyeLike ${MGL})
