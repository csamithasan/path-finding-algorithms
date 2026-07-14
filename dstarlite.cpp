#include "dstarlite.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>

extern GridWorld grid_world;

DStarLite::DStarLite(int rows_, int cols_, unsigned int heuristic_, std::string gridWorldName_)
: rows(rows_), cols(cols_), heuristic(heuristic_), gridWorldName(gridWorldName_) {
    maze.resize(rows, std::vector<vertex>(cols));
}

// Sync grid_world → maze
void DStarLite::syncFromGridWorld() {
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            maze[y][x] = grid_world.getVertex(y, x); // requires getVertex(row,col)
        }
    }
    vertex s = grid_world.getStartVertex();
    vertex g = grid_world.getGoalVertex();
    s_start = &maze[s.row][s.col];
    s_goal  = &maze[g.row][g.col];
}

// Sync maze → grid_world
void DStarLite::syncToGridWorld() {
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            grid_world.updateVertex(y, x, maze[y][x]); // requires updateVertex(row,col,vertex)
        }
    }
}

// heuristic between a node and the start (D* Lite runs from goal to start,
// so heuristic must estimate cost from node to start)
static double heuristicBetween(const vertex* a, const vertex* b, unsigned int heuristicType) {
    if (!a || !b) return 0.0;
    int dx = std::abs(a->col - b->col);
    int dy = std::abs(a->row - b->row);
    if (heuristicType == EUCLIDEAN) {
        return std::hypot((double)dx, (double)dy);
    } else { // CHEBYSHEV
        return (double)std::max(dx, dy);
    }
}

std::pair<double,double> DStarLite::calculateKey(vertex* u) const {
    double g_rhs = std::min(u->g, u->rhs);
    // h(u) is heuristic from u to start
    double h = heuristicBetween(u, s_start, heuristic);
    double k1 = g_rhs + h + km;
    double k2 = g_rhs;
    return {k1, k2};
}

void DStarLite::removeFromQueue(vertex* u) {
    auto it = inQueue.find(u);
    if (it == inQueue.end()) return;
    NodeKey nk(u, it->second.first, it->second.second);
    auto sit = U.find(nk);
    if (sit != U.end()) U.erase(sit);
    inQueue.erase(it);
}

void DStarLite::insertOrUpdate(vertex* u) {
    // compute key
    auto key = calculateKey(u);
    // if already in queue erase old
    removeFromQueue(u);
    // only insert if g != rhs
    if (std::abs(u->g - u->rhs) > 1e-12) {
        NodeKey nk(u, key.first, key.second);
        U.insert(nk);
        inQueue[u] = {key.first, key.second};
        if ((int)U.size() > max_queue) max_queue = (int)U.size();
    }
}

double DStarLite::stepCost(const vertex* u, const vertex* v) const {
    if (!u || !v) return INF_VAL();
    if (v->type == '1') return INF_VAL(); // blocked

    int dx = std::abs(u->col - v->col);
    int dy = std::abs(u->row - v->row);

    if (dx + dy == 1) return 1.0;             // straight
    if (dx == 1 && dy == 1) return std::sqrt(2.0); // diagonal

    return INF_VAL();
}

void DStarLite::getNeighbors(vertex* u, std::vector<vertex*>& out) const {
    static const int D[8][2] = {
        {1,0},{-1,0},{0,1},{0,-1},      // straight
        {1,1},{1,-1},{-1,1},{-1,-1}     // diagonals
    };
    out.clear();
    for (auto& d : D) {
        int nx = u->col + d[0];
        int ny = u->row + d[1];
        if (nx >= 0 && nx < cols && ny >= 0 && ny < rows) {
            out.push_back(const_cast<vertex*>(&maze[ny][nx]));
        }
    }
    vertex_access++;
}

void DStarLite::updateVertex(vertex* u) {
    if (u == nullptr) return;
    if (u != s_goal) {
        double min_rhs = INF_VAL();
        std::vector<vertex*> nbrs;
        getNeighbors(u, nbrs);
        for (auto* s : nbrs) {
            double c = stepCost(u, s); // cost(u,s)
            if (c < INF_VAL()) {
                double cand = s->g + c; // c + g(s)
                if (cand < min_rhs) min_rhs = cand;
            }
        }
        u->rhs = min_rhs;
    }
    insertOrUpdate(u);
    vertex_access++;
}

void DStarLite::computeShortestPath(bool fresh) {
    // Reset stats
    expansions = 0;
    path_length = 0;

    if (fresh) {
        // On the very first planning we must reset
        U.clear();
        inQueue.clear();
        km = 0.0;                 // (unless start moves in your experiments)
        insertOrUpdate(s_goal);   // must seed with goal
    }

    // Helper to compare keys
    auto keyLess = [&](const std::pair<double,double>& a, const std::pair<double,double>& b) {
        if (a.first < b.first - 1e-12) return true;
        if (a.first > b.first + 1e-12) return false;
        if (a.second < b.second - 1e-12) return true;
        if (a.second > b.second + 1e-12) return false;
        return false;
    };

    while (!U.empty()) {
        NodeKey top = *U.begin();
        vertex* u = top.v;
        std::pair<double,double> k_old = {top.k1, top.k2};
        std::pair<double,double> k_start = calculateKey(s_start);

        // stop condition
        if (!keyLess(k_old, k_start) && (std::abs(s_start->rhs - s_start->g) < 1e-12)) break;

        U.erase(U.begin());
        inQueue.erase(u);

        std::pair<double,double> k_new = calculateKey(u);

        if (keyLess(k_old, k_new)) {
            NodeKey nk(u, k_new.first, k_new.second);
            U.insert(nk);
            inQueue[u] = {k_new.first, k_new.second};
            continue;
        }

        expansions++;

        if (u->g > u->rhs) {
            u->g = u->rhs;
            std::vector<vertex*> preds;
            getNeighbors(u, preds);
            for (auto* p : preds) updateVertex(p);
        } else {
            u->g = INF_VAL();
            updateVertex(u);
            std::vector<vertex*> preds;
            getNeighbors(u, preds);
            for (auto* p : preds) updateVertex(p);
        }
    }

    buildTrace();
}


void DStarLite::buildTrace() {
    // Clear existing traces
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            maze[y][x].trace = nullptr;
        }
    }

    // Greedily follow best neighbor from start to goal using g-values
    vertex* cur = s_start;
    const int MAX_STEPS = rows * cols + 5;
    path_length = 0;

    for (int step = 0; step < MAX_STEPS && cur != nullptr && cur != s_goal; ++step) {
        std::vector<vertex*> nbrs;
        getNeighbors(cur, nbrs);

        vertex* best = nullptr;
        double bestCost = INF_VAL();

        for (auto* n : nbrs) {
            double c = stepCost(cur, n);
            if (c >= INF_VAL()) continue;
            // if neighbor's g is INF skip (not reachable)
            if (n->g >= INF_VAL()) continue;
            double cand = n->g + c; // cost to go via neighbor
            if (cand < bestCost) {
                bestCost = cand;
                best = n;
            }
        }

        if (!best) break;
        // Set forward trace (used by drawing)
        cur->trace = best;
        cur = best;
        path_length++;
    }
}

void DStarLite::initialise(int startX, int startY, int goalX, int goalY) {
    syncFromGridWorld();

    if (startY < 0 || startY >= rows || startX < 0 || startX >= cols) return;
    if (goalY < 0 || goalY >= rows || goalX < 0 || goalX >= cols) return;

    s_start = &maze[startY][startX];
    s_goal  = &maze[goalY][goalX];

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            maze[y][x].g   = INF_VAL();
            maze[y][x].rhs = INF_VAL();
            maze[y][x].trace = nullptr;
        }
    }
    s_goal->rhs = 0.0;
}

void DStarLite::run() {
    auto t0 = std::chrono::high_resolution_clock::now();
    computeShortestPath(true);   // initial planning
    auto t1 = std::chrono::high_resolution_clock::now();
    runtime_sec = std::chrono::duration<double>(t1 - t0).count();

    syncToGridWorld();
    grid_world.displayPath_for_dStarLite();
}

void DStarLite::finalPlanning() {
    // --- simulate environment change ---
    vertex* toBlock = nullptr;
    if (s_start && s_start->trace) {
        toBlock = s_start->trace;
        if (toBlock && toBlock != s_goal) {
            toBlock->type = '1'; // blocked
            toBlock->g = INF_VAL();
            toBlock->rhs = INF_VAL();
        }
    }

    // If we blocked a cell, we must update it + its neighbors
    if (toBlock) {
        updateVertex(toBlock);
        std::vector<vertex*> nbrs;
        getNeighbors(toBlock, nbrs);
        for (auto* n : nbrs) {
            updateVertex(n);
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    computeShortestPath(false);   // incremental replan
    auto t1 = std::chrono::high_resolution_clock::now();
    runtime_sec = std::chrono::duration<double>(t1 - t0).count();

    syncToGridWorld();
    grid_world.displayPath_for_dStarLite();

    std::cout << "D* Lite final planning.." << std::endl;
}



void DStarLite::printResults() {
    std::cout << "[D* Lite] path_length=" << path_length
              << ", expansions=" << expansions
              << ", vertex_accesses=" << vertex_access
              << ", max_queue=" << max_queue
              << ", runtime=" << runtime_sec << "s"
              << std::endl;
}

