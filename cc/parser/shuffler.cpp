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

unsigned long long shuffle_size;
unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
const std::string STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

struct Game {
    int winner;
    int welo;
    int belo;
    int tc;
    Game(const int win, const int we, const int be, const int t): winner(win), welo(we), belo(be), tc(t) {}
    bool operator==(Game& other) {
        return winner == other.winner && welo == other.welo && belo == other.belo && tc == other.tc;
    }
};

struct Position {
    std::string fen;
    float eval, eval_next;
    int time_taken;
    Game game;
    Position(const std::string& s, const float ev, const Game& g, const int time_taken): fen(s), eval(ev), eval_next(ev), game(g), time_taken(time_taken) {}
    std::string to_string() {
        std::stringstream ss;
        ss << fen << ", " << eval << ", " << eval_next << ", " << game.winner << ", " << game.welo << ", " << game.belo << ", " << game.tc << ", " << time_taken << '\n';
        return ss.str();
    }
};


struct Data {
    std::ifstream file;
    Game current_game;
    Position current_pos;
    unsigned long long i;
    std::vector<Position> positions;
    bool last = false;
    Data(const fs::path& f): file(f), current_game(next_game()), current_pos(STARTPOS, 0, current_game, 0), i(0), positions(populate_positions()) {}

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

    void next_position() {
        last = false;
        std::string fen;
        getline(file, fen);
        if (file.peek() == EOF) {
            last = true;
            return;
        }
        if (fen.size() < 5) {
            current_game = next_game();
            last = true;
            return;
        }
        std::string line;
        getline(file, line);
        float eval = std::stof(line);
        getline(file, line);
        int time_taken = std::stoi(line);
        current_pos = Position(fen, eval, current_game, time_taken);
    }

    std::vector<Position> populate_positions() {
        std::vector<Position> pos;
        while ((pos.size() < shuffle_size || !last) && file.peek() != EOF) {
            next_position();
            if (last) {
                pos.pop_back();
                continue;
            }
            pos.push_back(current_pos);
        }
        for (auto it = pos.begin()+1; it != pos.end(); ++it) {
            if (it->game == (it-1)->game)
                (it-1)->eval_next = it->eval;
        }
        std::shuffle(pos.begin(), pos.end(), std::default_random_engine(seed));
        i = 0;
        return pos;
    }

    std::string next() {
        if (i >= positions.size()) positions = populate_positions();
        return positions[i++].to_string();
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
    shuffle_size = std::atoll(argv[4]);

    std::vector<Data> file_vec = build_file_vec(in_path);
    std::shuffle(file_vec.begin(), file_vec.end(), std::default_random_engine(seed));

    std::ofstream out_file{out_path + "shuffled_0.data"};
    unsigned long long current_pos_count = 0;
    unsigned long long pos_count = 0;

    bool done = false;
    while (!done) {
        done = true;
        for (auto it = file_vec.begin(); it != file_vec.end(); ++it) {
            if (it->file.peek() == EOF) continue;
            if (current_pos_count >= positions_per_file) {
                out_file.close();
                out_file.clear();
                out_file.open(out_path + "shuffled_" + std::to_string(pos_count) + ".data");
                current_pos_count = 0;
            }
            out_file << it->next();
            ++current_pos_count;
            ++pos_count;
            done = false;
        }
    }
    return 0;
}
