#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <unistd.h>

// Filters out games with no evaluation and useless headers, and optionally splits output into multiple files.
// assumes that the movelist is in a single line

std::unordered_set<std::string> keep_headers{"[Result", "[WhiteElo", "[BlackElo", "[TimeControl"};
long long game_count = 0;
long long current_game_count = 0;

struct Game {
    std::string movelist;
    std::vector<std::string> headers;
    void add_header(const std::string& line) {
        std::string header_name = line.substr(0, line.find_first_of(' '));
        if (keep_headers.find(header_name) != keep_headers.end()) {
            headers.push_back(line);
        }
    }
};

void validate_and_write(const Game& game, std::ofstream& of) {
    // Checks if game contains evaluations and if so, saves it to the of stream.
    // We can assume that the eval qualifier will be present in the first 20 characters of the movelist, for simplicity.
    std::string subs = game.movelist.substr(0, 20);
    if (subs.find("[%eval") == std::string::npos) return;
    for (auto &header : game.headers) {
        of << header << '\n';
    }
    of << '\n';
    of << game.movelist;
    of << "\n\n";

    game_count += 1;
    current_game_count += 1;
}


int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Wrong arguments!\nargs: path-to-pgn path-to-output games-per-file (0 to disable splitting)\n";
        exit(1);
    }
    std::ifstream f{argv[1]};
    std::ofstream out_file{std::string(argv[2]) + "_0.pgn"};
    unsigned long long games_per_file = std::atoll(argv[3]);
    if (games_per_file == 0) games_per_file = -1;

    std::string s;
    Game current;
    long long lines = 0;
    while (f) {
        std::getline(f, s);
        if (s.substr(0, 3) == "1-0" || s.substr(0, 3) == "0-1" || s.substr(0, 1) == "*") {
        	current = Game();
        	continue;
        }
        if (s[0] != '[' && !std::isspace(s[0]) && s[0] != '\0' && s[0] != '1') {
            std::cerr << "line must start with [, \\0 or 1 \nerror in line " << lines << " with content:\n" << s << '\n';
            exit(1);
        }
        if (s[0] == '[') current.add_header(s);
        else if (s[0] == '1') {
            current.movelist = s;
            validate_and_write(current, out_file);
            current = Game();
            if (current_game_count >= games_per_file) {
                out_file.close();
                out_file.clear();
                out_file.open(std::string(argv[2]) + "_" + std::to_string(game_count) + ".pgn");
                current_game_count = 0;
            }
        }
        ++lines;
    }
    std::cout << game_count << '\n';
    return 0;
}
