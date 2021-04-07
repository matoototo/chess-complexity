#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Wrong arguments!\nargs: path-to-pgn\n";
        exit(1);
    }
    std::ifstream f{argv[1]};
    std::string s;
    int lines = 0;
    while (f) {
        std::getline(f, s);
        ++lines;
    }
    std::cout << lines << '\n';
    return 0;
}
