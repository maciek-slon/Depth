# Include the directory itself as a path to include directories
SET(CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a variable containing all .cpp files:
FILE(GLOB files *.cpp)

# Create an executable file from sources:
ADD_LIBRARY(NormalEstimator SHARED ${files})

# Link external libraries
# TARGET_LINK_LIBRARIES(NormalEstimator ${DisCODe_LIBRARIES} )

# Add aother libraries if necessary, for example:
TARGET_LINK_LIBRARIES(NormalEstimator ${DisCODe_LIBRARIES} ${OpenCV_LIBS} )

INSTALL_COMPONENT(NormalEstimator)
