#include <algorithm>
#include <chrono>
#include <random>
#include <filesystem>
namespace fs = std::filesystem;

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <unistd.h>

struct Game {
    int winner;
    int welo;
    int belo;
    int tc;
    Game(const int& win, const int& we, const int& be, const int& t): winner(win), welo(we), belo(be), tc(t) {}
};

struct Position {
    std::string fen;
    float eval;
    Game game;
    Position(const std::string& s, const float& ev, const Game& g): fen(s), eval(ev), game(g) {}
    std::string to_string() {
        std::stringstream ss;
        ss << fen << ", " << eval << ", " << game.winner << ", " << game.welo << ", " << game.belo << ", " << game.tc << '\n';
        return ss.str();
    }
};


struct Data {
    std::ifstream file;
    Game current_game;

    Data(const fs::path& f): file(f), current_game(next_game()) {}

    Game next_game() {
        std::string line;
        getline(file, line);
        int winner = std::stoi(line);
        getline(file, line);
        int welo = std::stoi(line);
        getline(file, line);
        int belo = std::stoi(line);
        getline(file, line);
        int tc = std::stoi(line);
        return Game(winner, welo, belo, tc);
    }

    Position next_position() {
        std::string fen;
        getline(file, fen);
        if (fen.size() < 10) {
            current_game = next_game();
            return next_position();
        }
        std::string line;
        getline(file, line);
        int eval = std::stof(line);
        return Position(fen, eval, current_game);
    }

};

std::vector<Data> build_file_vec(const std::string& path) {
    std::vector<Data> vec;
    for (const auto& file : fs::recursive_directory_iterator(path))
        if (file.path().extension() == ".data") vec.push_back(Data(file.path()));
    return vec;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Wrong arguments!\nargs: path-to-data path-to-output positions-per-file shuffle-size\n";
        exit(1);
    }
    std::string in_path{argv[1]};
    std::string out_path{argv[2]};
    unsigned long long positions_per_file = std::atoll(argv[3]);
    unsigned long long shuffle_size = std::atoll(argv[4]);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::vector<Data> file_vec = build_file_vec(in_path);
    std::shuffle(file_vec.begin(), file_vec.end(), std::default_random_engine(seed));

    std::ofstream out_file{out_path + "shuffled_0.data"};
    unsigned long long current_pos_count = 0;
    unsigned long long pos_count = 0;

    while (true) {
        for (auto it = file_vec.begin(); it != file_vec.end(); ++it) {
            if (!it->file) continue;
            if (current_pos_count >= positions_per_file) {
                out_file.close();
                out_file.clear();
                out_file.open(out_path + "shuffled_" + std::to_string(pos_count) + ".data");
                current_pos_count = 0;
            }
            out_file << it->next_position().to_string();
            ++current_pos_count;
            ++pos_count;
        }
    }
    return 0;
}
