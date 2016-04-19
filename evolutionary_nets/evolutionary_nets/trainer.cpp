#include "trainer.h"

unsigned int Trainer::generate_random_integer_between_range(unsigned int min, unsigned int max) {
    return min + ( std::rand() % ( max - min + 1 ) );
}

























