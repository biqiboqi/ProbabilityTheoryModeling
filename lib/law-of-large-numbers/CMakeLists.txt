add_library(law-of-large-numbers STATIC
        LawOfLargeNumbersSimulator.cpp
)

target_link_libraries(law-of-large-numbers PUBLIC distributions)