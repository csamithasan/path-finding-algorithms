#include "lpastar.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <chrono>
#include "gridworld.h"
extern GridWorld grid_world;

// ------------------------------
// LPA* implementation aligned to project's globals/renderer
// - Uses rhs(start)=0 and propagates to goal
// - Heuristic is distance to goal
// - Marks path by setting type=5 and also fills trace pointers
// ------------------------------

LpaStar::LpaStar(int rows_, int cols_, unsigned int _heuristic, std::string _gridWorldName)
    : rows(rows_), cols(cols_), heuristic(_heuristic), gridWorldName(std::move(_gridWorldName)) {
    maze.resize(rows, std::vector<LpaStarCell>(cols));
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            auto &c = maze[y][x];
            c.x = x;
            c.y = y;
            c.type = 0;
            c.g = std::numeric_limits<double>::infinity();
            c.rhs = std::numeric_limits<double>::infinity();
            c.h = 0.0;
            c.key[0] = c.key[1] = std::numeric_limits<double>::infinity();
            c.parent = nullptr;
            c.trace = nullptr;
            for (int d=0; d<DIRECTIONS; ++d) {
                c.move[d] = nullptr;
                c.predecessor[d] = nullptr;
                c.linkCost[d] = 1.0;
            }
            c.obstacle = 0;
            c.generated = 0;
            c.heapindex = -1;
        }
    }
}

void LpaStar::syncFromGridWorld() {
    // pull obstacle layout and start/goal from the renderer's map
    const int H = std::min(rows, (int)grid_world.map.size());
    const int W = H ? std::min(cols, (int)grid_world.map[0].size()) : 0;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            // GridWorld stores chars: '0' traversable, '1' blocked, etc.
            // We only need passable vs blocked here.
            maze[y][x].type = (grid_world.map[y][x].type == '1') ? 1 : 0;
        }
    }

    // keep start/goal in sync with what the UI uses
    const int sy = grid_world.startVertex.row;
    const int sx = grid_world.startVertex.col;
    const int gy = grid_world.goalVertex.row;
    const int gx = grid_world.goalVertex.col;

    if (sy >= 0 && sy < rows && sx >= 0 && sx < cols) start = &maze[sy][sx];
    if (gy >= 0 && gy < rows && gx >= 0 && gx < cols) goal  = &maze[gy][gx];
}

void LpaStar::syncToGridWorld() {
    // push numeric fields back so the drawer (which reads grid_world.map) can trace the path
    const int H = std::min(rows, (int)grid_world.map.size());
    const int W = H ? std::min(cols, (int)grid_world.map[0].size()) : 0;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            auto &dst = grid_world.map[y][x];
            const auto &src = maze[y][x];

            dst.g       = src.g;
            dst.rhs     = src.rhs;
            dst.h       = src.h;
            dst.key[0]  = src.key[0];
            dst.key[1]  = src.key[1];
            // do NOT overwrite dst.type here (renderer colors use it)
            // do NOT touch dst.move/dst.linkCost (renderer owns those)
        }
    }
}


void LpaStar::initialise(int sx, int sy, int gx, int gy) {
    U.clear();
    stateExpansions = 0;
    vertexAccesses = 0;
    pathLength = 0.0;
    runningTime = 0.0;
    maxQueueSize = 0;

    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) {
            auto &c = maze[y][x];
            c.g = std::numeric_limits<double>::infinity();
            c.rhs = std::numeric_limits<double>::infinity();
            c.h = 0.0;
            c.key[0] = c.key[1] = std::numeric_limits<double>::infinity();
            c.trace = nullptr;
            c.parent = nullptr;
            if (c.type == 5) c.type = 0;
        }

    // Bounds guard — prevents OOB when taking &maze[sy][sx]
    if (sx < 0 || sx >= cols || sy < 0 || sy >= rows ||
        gx < 0 || gx >= cols || gy < 0 || gy >= rows) {
        std::cerr << "LPA*: initialise(): start/goal indices out of bounds\n";
        start = goal = nullptr;
        return;
    }

    start = &maze[sy][sx];
    goal  = &maze[gy][gx];

    // Heuristics/keys depend on goal, but we’ll re-sync before planning anyway
    updateHValues();
    updateAllKeyValues();

    start->rhs = 0.0;
    insertOrUpdate(start);

    episode = 0;
    strEpisode = "initial_planning";

    // (ok to keep this; we also sync at start of each planning pass)
    syncFromGridWorld();
}

double LpaStar::heuristicDist(const LpaStarCell& a, const LpaStarCell& b) const {
    if (heuristic == EUCLIDEAN) {
        double dx = double(a.x - b.x);
        double dy = double(a.y - b.y);
        return std::sqrt(dx*dx + dy*dy);
    } else { // CHEBYSHEV
        return double(std::max(std::abs(a.x - b.x), std::abs(a.y - b.y)));
    }
}

double LpaStar::stepCost(const LpaStarCell& /*a*/, const LpaStarCell& /*b*/) const {
    // uniform cost for simplicity; adjust if diagonals should be sqrt(2)
    return 1.0;
}

std::pair<double,double> LpaStar::calcKey(LpaStarCell* u) const {
    const double min_g_rhs = std::min(u->g, u->rhs);
    return { min_g_rhs + heuristicDist(*u, *goal), min_g_rhs };
}

void LpaStar::removeIfPresent(LpaStarCell* u) {
    auto it = U.find(u);
    if (it != U.end()) U.erase(it);
}

void LpaStar::insertOrUpdate(LpaStarCell* u) {
    if (u->g != u->rhs) {
        auto k = calcKey(u);
        U[u] = k;
        u->key[0] = k.first;
        u->key[1] = k.second;

        if ((int)U.size() > maxQueueSize) {
            maxQueueSize = (int)U.size();
        }
    } else {
        removeIfPresent(u);
        u->key[0] = u->key[1] = std::numeric_limits<double>::infinity();
    }
}

std::vector<LpaStarCell*> LpaStar::getNeighbors(LpaStarCell* u) {
    static const int D[4][2] = {
        {1,0},{-1,0},{0,1},{0,-1}
    };
    std::vector<LpaStarCell*> out;
    out.reserve(4);
    for (auto &d : D) {
        int nx = u->x + d[0];
        int ny = u->y + d[1];
        if (nx >= 0 && nx < cols && ny >= 0 && ny < rows) {
            auto &c = maze[ny][nx];
            if (c.type != 1) out.push_back(&c); // 1 == blocked
        }
    }
    vertexAccesses += out.size();
    return out;
}


void LpaStar::updateVertex(LpaStarCell* u, int /*phase*/) {
    if (u != start) {
        double min_rhs = std::numeric_limits<double>::infinity();
        for (auto* s : getNeighbors(u)) {
            min_rhs = std::min(min_rhs, s->g + stepCost(*u, *s));
        }
        u->rhs = min_rhs;
    }
    insertOrUpdate(u);
}

void LpaStar::computeShortestPath(int /*phase*/) {
    if (!start || !goal) {
        std::cerr << "LPA*: computeShortestPath(): start/goal not set\n";
        return;
    }

    while (!U.empty()) {
        auto it = std::min_element(U.begin(), U.end(),
            [](const auto& a, const auto& b) {
                if (a.second.first == b.second.first)
                    return a.second.second < b.second.second;
                return a.second.first < b.second.first;
            });

        LpaStarCell* u = it->first;
        auto k_old = it->second;
        auto k_new = calcKey(u);

        // Guard: goal must be valid here
        auto curGoalKey = calcKey(goal);
        if (!(k_old < curGoalKey || goal->g != goal->rhs)) break;

        U.erase(it);
        stateExpansions++;

        if (k_old < k_new) {
            U[u] = k_new;
            u->key[0] = k_new.first;
            u->key[1] = k_new.second;
        } else if (u->g > u->rhs) {
            u->g = u->rhs;
            for (auto* s : getNeighbors(u)) updateVertex(s, 0);
        } else {
            u->g = std::numeric_limits<double>::infinity();
            updateVertex(u, 0);
            for (auto* s : getNeighbors(u)) updateVertex(s, 0);
        }
    }
}

void LpaStar::clearPathPaint() {
    for (auto &row : maze) {
        for (auto &c : row) {
            if (c.type == 5) c.type = 0;
            c.trace = nullptr;
        }
    }
}

void LpaStar::markPathBacktrack() {
    grid_world.clearPathTraces();

    if (!start || !goal) return;
    if (!std::isfinite(goal->g)) return;

    // Renderer dimensions to guard map writes
    const int H = (int)grid_world.map.size();
    const int W = H ? (int)grid_world.map[0].size() : 0;
    if (H == 0 || W == 0) return;

    auto in_bounds = [&](int y, int x) {
        return (y >= 0 && y < H && x >= 0 && x < W);
    };

    LpaStarCell* cur  = goal;
    const int maxSteps = rows * cols + 10;
    int stepsOnPath = 0;

    // Always terminate the goal link inside bounds
    if (in_bounds(goal->y, goal->x)) {
        grid_world.map[goal->y][goal->x].trace = nullptr;
    }

    for (int step = 0; step < maxSteps && cur != start; ++step) {
        LpaStarCell* best = nullptr;
        double bestCost = std::numeric_limits<double>::infinity();

        for (auto* s : getNeighbors(cur)) {
            if (!s || !std::isfinite(s->g)) continue;
            double c = s->g + stepCost(*cur, *s);
            if (c < bestCost) { bestCost = c; best = s; }
        }

        if (!best) return; // no predecessor—give up safely

        // Only write the forward link if both endpoints are inside the renderer map
        if (in_bounds(best->y, best->x) && in_bounds(cur->y, cur->x)) {
            grid_world.map[best->y][best->x].trace = &grid_world.map[cur->y][cur->x];
        } else {
            // bail if mapping is inconsistent with the renderer grid
            return;
        }

        cur = best;
        ++stepsOnPath;
    }

    // Optional: store unit step count
    pathLength = stepsOnPath;
}





void LpaStar::initialPlanning() {
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            maze[y][x].trace = nullptr;

    auto t0 = std::chrono::high_resolution_clock::now();

    episode = 0;
    strEpisode = "initial_planning";

    // keep this first
    syncFromGridWorld();

    if (!start || !goal) {
        std::cerr << "LPA*: initialPlanning(): start/goal missing\n";
        return;
    }

    updateHValues();
    updateAllKeyValues();

    insertOrUpdate(start); // rhs(start)=0 was set in initialise()
    computeShortestPath(episode);

    markPathBacktrack();

    // Draw only if start has a forward link or start==goal
    const bool can_draw =
        (grid_world.startVertex.row >= 0 && grid_world.startVertex.col >= 0 &&
         grid_world.startVertex.row < (int)grid_world.map.size() &&
         grid_world.startVertex.col < (int)grid_world.map[0].size() &&
         (grid_world.map[grid_world.startVertex.row][grid_world.startVertex.col].trace != nullptr ||
          (start == goal)));

    if (can_draw) {
        grid_world.displayPath_for_lpaStar();
    } else {
        std::cerr << "LPA*: no valid trace chain to draw (start->trace is null)\n";
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    runningTime = std::chrono::duration<double>(t1 - t0).count();

    syncToGridWorld();
}

void LpaStar::finalPlanning() {
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            maze[y][x].trace = nullptr;

    auto t0 = std::chrono::high_resolution_clock::now();

    episode = 1;
    strEpisode = "final_planning";

    syncFromGridWorld();

    if (!start || !goal) {
        std::cerr << "LPA*: finalPlanning(): start/goal missing\n";
        return;
    }

    updateHValues();
    updateAllKeyValues();

    computeShortestPath(episode);
    markPathBacktrack();

    const bool can_draw =
        (grid_world.startVertex.row >= 0 && grid_world.startVertex.col >= 0 &&
         grid_world.startVertex.row < (int)grid_world.map.size() &&
         grid_world.startVertex.col < (int)grid_world.map[0].size() &&
         (grid_world.map[grid_world.startVertex.row][grid_world.startVertex.col].trace != nullptr ||
          (start == goal)));

    if (can_draw) {
        grid_world.displayPath_for_lpaStar();
    } else {
        std::cerr << "LPA*: no valid trace chain to draw (start->trace is null)\n";
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    runningTime = std::chrono::duration<double>(t1 - t0).count();

    syncToGridWorld();
}

void LpaStar::updateHValues() {
    if (!start || !goal) return;
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            maze[y][x].h = heuristicDist(maze[y][x], *goal);
        }
    }
}

void LpaStar::updateAllKeyValues() {
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            auto* u = &maze[y][x];
            auto k = calcKey(u);
            u->key[0] = k.first;
            u->key[1] = k.second;
        }
    }
}

void LpaStar::printResults() {
    std::cout << "[LPA* " << strEpisode << "] "
              << "Expansions=" << stateExpansions
              << ", MaxQueue=" << maxQueueSize
              << ", VertexAccesses=" << vertexAccesses
              << ", PathLength=" << pathLength
              << ", Runtime=" << runningTime << "s"
              << std::endl;
}
