# -*- coding: utf-8 -*-
"""RouteOptimization_test1_improved_v2.py

Improved CVRP solver v2 — adds:
  • Auto-calculate min vehicles when not in file
  • Sweep algorithm (angle-based construction)
  • Perturbation to escape local optima
  • Performance-capped cross exchange
  • Generalized partition for k vehicles (small instances)
  • Time limit to prevent very long runs
"""

# ============================================================
#  CVRP — Multi-Strategy Solver v2 (Google Colab / Local)
# ============================================================

import math
import re
import os
import sys
import random
import time
from collections import Counter
import pandas as pd

# ==================== ตั้งค่า ====================
max_vehicles = None       # จะอ่านจากไฟล์ .vrp เท่านั้น
TIME_LIMIT = 120          # วินาที — หยุดค้นหาเมื่อเกินเวลา
PERTURB_ITER = 50         # จำนวนรอบ perturbation
random.seed(42)           # ให้ผลลัพธ์ซ้ำได้

# ==================== 1. อ่านไฟล์ .vrp ====================
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("กรุณาเลือกไฟล์ .vrp จากเครื่องของคุณ\n")
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
    content = uploaded[filename].decode("utf-8")
else:
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        vrp_files = [f for f in os.listdir('.') if f.endswith('.vrp')]
        if not vrp_files:
            for d in ['Dataset', '0 CVRP Benchmark']:
                if os.path.isdir(d):
                    vrp_files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.vrp')]
                    if vrp_files:
                        break
        if not vrp_files:
            print("ไม่พบไฟล์ .vrp — กรุณาระบุ path เป็น argument")
            sys.exit(1)
        filepath = vrp_files[0]
    filename = os.path.basename(filepath)
    with open(filepath, 'r') as f:
        content = f.read()

raw_lines = content.strip().split("\n")

# ==================== 2. Parser ====================
def parse_vrp_file(raw_lines):
    """อ่านข้อมูลจากไฟล์ .vrp แล้วแยกเป็น metadata, พิกัด, demand, depot"""
    metadata = {}
    node_coords = {}
    demands = {}
    depots = []
    section = None

    for line in raw_lines:
        line = line.strip()
        if not line or line == "EOF":
            if line == "EOF":
                break
            continue

        if line == "NODE_COORD_SECTION":
            section = "COORD"; continue
        elif line == "DEMAND_SECTION":
            section = "DEMAND"; continue
        elif line == "DEPOT_SECTION":
            section = "DEPOT"; continue

        if section == "COORD":
            parts = line.split()
            node_coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
        elif section == "DEMAND":
            parts = line.split()
            demands[int(parts[0])] = int(parts[1])
        elif section == "DEPOT":
            val = int(line)
            if val == -1:
                section = None
            else:
                depots.append(val)
        else:
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()

    return metadata, node_coords, demands, depots


metadata, node_coords, demands, depots = parse_vrp_file(raw_lines)

# ==================== 3. อ่านจำนวนรถจากไฟล์ ====================
vehicles_source = ""
comment_text = metadata.get("COMMENT", "")
if comment_text:
    match = re.search(r"No of trucks:\s*(\d+)", comment_text, re.IGNORECASE)
    if match:
        max_vehicles = int(match.group(1))
        vehicles_source = "COMMENT ในไฟล์"

# ==================== 4. Validation ====================
file_type = metadata.get("TYPE", "")
if file_type != "CVRP":
    raise ValueError(f"หยุด: TYPE = '{file_type}' — รองรับเฉพาะ CVRP")

ewt = metadata.get("EDGE_WEIGHT_TYPE", "")
if ewt != "EUC_2D":
    raise ValueError(f"หยุด: EDGE_WEIGHT_TYPE = '{ewt}' — รองรับเฉพาะ EUC_2D")

if len(depots) != 1:
    raise ValueError(f"หยุด: พบ Depot {len(depots)} จุด ({depots}) — รองรับ Depot เดียว")

depot = depots[0]

if depot not in node_coords:
    raise ValueError(f"หยุด: Depot (Node {depot}) ไม่พบใน NODE_COORD_SECTION")
if depot not in demands:
    raise ValueError(f"หยุด: Depot (Node {depot}) ไม่พบใน DEMAND_SECTION")

dimension = int(metadata.get("DIMENSION", 0))
if len(node_coords) != dimension:
    raise ValueError(f"หยุด: DIMENSION = {dimension} แต่ NODE_COORD_SECTION มี {len(node_coords)} จุด")
if len(demands) != dimension:
    raise ValueError(f"หยุด: DIMENSION = {dimension} แต่ DEMAND_SECTION มี {len(demands)} จุด")

if set(node_coords.keys()) != set(demands.keys()):
    raise ValueError("หยุด: Node ID ใน NODE_COORD_SECTION ไม่ตรงกับ DEMAND_SECTION")

capacity = int(metadata.get("CAPACITY", 0))
customers = sorted([n for n in node_coords if n != depot])
total_demand = sum(demands[c] for c in customers)

# Auto-calculate minimum vehicles if file didn't specify
min_vehicles_needed = math.ceil(total_demand / capacity)
if max_vehicles is None:
    max_vehicles = min_vehicles_needed
    vehicles_source = f"คำนวณอัตโนมัติ (ceil({total_demand}/{capacity}))"

# ==================== 5. สรุปข้อมูล ====================
print("\n" + "=" * 55)
print("  สรุปข้อมูลที่อ่านได้จากไฟล์")
print("=" * 55)
print(f"  ชื่อไฟล์          : {filename}")
print(f"  Depot             : Node {depot}")
print(f"  จำนวน Customer    : {len(customers)}")
print(f"  Capacity          : {capacity}")
print(f"  max_vehicles      : {max_vehicles}  (จาก: {vehicles_source})")
print(f"  min vehicles need : {min_vehicles_needed}")
print(f"  Demand รวม        : {total_demand}")
print(f"  EDGE_WEIGHT_TYPE  : {ewt}")

# ==================== 6. Distance ====================
def euc_2d(coord1, coord2):
    """คำนวณระยะทาง EUC_2D แล้วปัดด้วย floor(distance + 0.5)"""
    dx = coord1[0] - coord2[0]
    dy = coord1[1] - coord2[1]
    raw = math.sqrt(dx * dx + dy * dy)
    return int(math.floor(raw + 0.5))

all_nodes = sorted(node_coords.keys())
dist_matrix = {}
for i in all_nodes:
    for j in all_nodes:
        dist_matrix[(i, j)] = euc_2d(node_coords[i], node_coords[j])

def route_distance(route):
    """คำนวณระยะทางรวมของ route"""
    return sum(dist_matrix[(route[i], route[i+1])] for i in range(len(route) - 1))

def route_load(route):
    """คำนวณ demand รวมของ route (ไม่นับ depot)"""
    return sum(demands[n] for n in route if n != depot)

def total_distance(routes):
    """คำนวณระยะทางรวมของทุก routes"""
    return sum(route_distance(r) for r in routes)


# ============================================================
#  7. Construction Heuristics
# ============================================================

def nearest_neighbor(depot, customers, max_vehicles):
    """Nearest Neighbor แบบเดิม"""
    unvisited = set(customers)
    routes = []
    for v in range(max_vehicles):
        if not unvisited:
            break
        route = [depot]
        current = depot
        load = 0
        while True:
            candidates = []
            for c in unvisited:
                if load + demands[c] <= capacity:
                    d = dist_matrix[(current, c)]
                    candidates.append((d, c))
            if not candidates:
                break
            candidates.sort(key=lambda x: (x[0], x[1]))
            chosen_dist, chosen = candidates[0]
            route.append(chosen)
            load += demands[chosen]
            current = chosen
            unvisited.remove(chosen)
        route.append(depot)
        routes.append(route)
    return routes, unvisited


def nearest_neighbor_seed(depot, customers, max_vehicles, seed):
    """Nearest Neighbor เริ่มจาก Customer ที่กำหนด (seed) สำหรับรถคันแรก"""
    unvisited = set(customers)
    routes = []
    for v in range(max_vehicles):
        if not unvisited:
            break
        route = [depot]
        current = depot
        load = 0
        if v == 0 and seed in unvisited and load + demands[seed] <= capacity:
            route.append(seed)
            load += demands[seed]
            current = seed
            unvisited.remove(seed)
        while True:
            candidates = []
            for c in unvisited:
                if load + demands[c] <= capacity:
                    d = dist_matrix[(current, c)]
                    candidates.append((d, c))
            if not candidates:
                break
            candidates.sort(key=lambda x: (x[0], x[1]))
            chosen_dist, chosen = candidates[0]
            route.append(chosen)
            load += demands[chosen]
            current = chosen
            unvisited.remove(chosen)
        route.append(depot)
        routes.append(route)
    return routes, unvisited


def clarke_wright_savings(depot, customers, max_vehicles):
    """Clarke-Wright Savings Algorithm"""
    savings_list = []
    for i in customers:
        for j in customers:
            if i < j:
                s = dist_matrix[(depot, i)] + dist_matrix[(depot, j)] - dist_matrix[(i, j)]
                savings_list.append((s, i, j))
    savings_list.sort(reverse=True)

    cw_routes = [[depot, c, depot] for c in customers]
    cw_demands = [demands[c] for c in customers]

    def find_route_idx(node):
        for idx, r in enumerate(cw_routes):
            if node in r[1:-1]:
                return idx
        return -1

    for s, i, j in savings_list:
        ri = find_route_idx(i)
        rj = find_route_idx(j)
        if ri == -1 or rj == -1 or ri == rj:
            continue
        if cw_demands[ri] + cw_demands[rj] > capacity:
            continue

        r_i = cw_routes[ri]
        r_j = cw_routes[rj]

        merged = None
        if r_i[-2] == i and r_j[1] == j:
            merged = r_i[:-1] + r_j[1:]
        elif r_j[-2] == j and r_i[1] == i:
            merged = r_j[:-1] + r_i[1:]
        elif r_i[-2] == i and r_j[-2] == j:
            merged = r_i[:-1] + r_j[-2:0:-1] + [depot]
        elif r_i[1] == i and r_j[1] == j:
            merged = [depot] + r_i[-2:0:-1] + r_j[1:]

        if merged:
            new_demand = cw_demands[ri] + cw_demands[rj]
            if ri > rj:
                cw_routes.pop(ri); cw_demands.pop(ri)
                cw_routes.pop(rj); cw_demands.pop(rj)
            else:
                cw_routes.pop(rj); cw_demands.pop(rj)
                cw_routes.pop(ri); cw_demands.pop(ri)
            cw_routes.append(merged)
            cw_demands.append(new_demand)

    # ตรวจว่าจำนวน route ไม่เกิน max_vehicles
    remaining = set()
    if len(cw_routes) > max_vehicles:
        while len(cw_routes) > max_vehicles:
            cw_routes.sort(key=lambda r: route_load(r))
            smallest = cw_routes.pop(0)
            merged_any = False
            for node in smallest[1:-1]:
                best_idx = -1
                best_pos = -1
                best_increase = float('inf')
                for idx, r in enumerate(cw_routes):
                    if route_load(r) + demands[node] > capacity:
                        continue
                    for pos in range(1, len(r)):
                        increase = (dist_matrix[(r[pos-1], node)] +
                                    dist_matrix[(node, r[pos])] -
                                    dist_matrix[(r[pos-1], r[pos])])
                        if increase < best_increase:
                            best_increase = increase
                            best_idx = idx
                            best_pos = pos
                if best_idx >= 0:
                    cw_routes[best_idx] = (cw_routes[best_idx][:best_pos] +
                                           [node] +
                                           cw_routes[best_idx][best_pos:])
                    merged_any = True
                else:
                    remaining.add(node)
            if not merged_any:
                for node in smallest[1:-1]:
                    remaining.add(node)

    return cw_routes, remaining


# [NEW] Sweep Algorithm — angle-based construction
def sweep_algorithm(depot, customers, max_vehicles, start_angle=0.0):
    """Sweep: เรียง Customer ตามมุมรอบ Depot แล้วกวาดใส่รถทีละคัน"""
    dx0, dy0 = node_coords[depot]

    # คำนวณมุมของแต่ละ Customer เทียบกับ Depot
    angles = []
    for c in customers:
        cx, cy = node_coords[c]
        angle = math.atan2(cy - dy0, cx - dx0)
        # ปรับมุมให้เริ่มจาก start_angle
        adjusted = (angle - start_angle) % (2 * math.pi)
        angles.append((adjusted, c))
    angles.sort()

    routes = []
    current_route = [depot]
    current_load = 0

    for _, c in angles:
        if current_load + demands[c] <= capacity:
            current_route.append(c)
            current_load += demands[c]
        else:
            current_route.append(depot)
            routes.append(current_route)
            current_route = [depot, c]
            current_load = demands[c]

    current_route.append(depot)
    routes.append(current_route)

    # ตรวจ: ถ้าเกิน max_vehicles ต้องรวม route เล็กๆ
    remaining = set()
    if len(routes) > max_vehicles:
        routes.sort(key=lambda r: route_load(r), reverse=True)
        keep = routes[:max_vehicles]
        extras = routes[max_vehicles:]
        for er in extras:
            for node in er[1:-1]:
                inserted = False
                best_idx = -1
                best_pos = -1
                best_increase = float('inf')
                for idx, r in enumerate(keep):
                    if route_load(r) + demands[node] > capacity:
                        continue
                    for pos in range(1, len(r)):
                        increase = (dist_matrix[(r[pos-1], node)] +
                                    dist_matrix[(node, r[pos])] -
                                    dist_matrix[(r[pos-1], r[pos])])
                        if increase < best_increase:
                            best_increase = increase
                            best_idx = idx
                            best_pos = pos
                if best_idx >= 0:
                    keep[best_idx] = (keep[best_idx][:best_pos] +
                                      [node] +
                                      keep[best_idx][best_pos:])
                else:
                    remaining.add(node)
        routes = keep

    return routes, remaining


# ============================================================
#  8. Local Search Improvements
# ============================================================

def two_opt(route):
    """2-opt: สลับลำดับ segment ภายใน route"""
    improved = True
    best = route[:]
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                d_old = dist_matrix[(best[i-1], best[i])] + dist_matrix[(best[j], best[j+1])]
                d_new = dist_matrix[(best[i-1], best[j])] + dist_matrix[(best[i], best[j+1])]
                if d_new < d_old:
                    best[i:j+1] = best[i:j+1][::-1]
                    improved = True
    return best


def or_opt(route):
    """Or-opt: ย้าย segment 1-3 nodes ไปตำแหน่งอื่นใน route"""
    improved = True
    best = route[:]
    best_dist = route_distance(best)
    while improved:
        improved = False
        for seg_len in [1, 2, 3]:
            for i in range(1, len(best) - seg_len - 1):
                segment = best[i:i+seg_len]
                remaining = best[:i] + best[i+seg_len:]
                for j in range(1, len(remaining)):
                    new_route = remaining[:j] + segment + remaining[j:]
                    nd = route_distance(new_route)
                    if nd < best_dist:
                        best = new_route
                        best_dist = nd
                        improved = True
                    new_route2 = remaining[:j] + segment[::-1] + remaining[j:]
                    nd2 = route_distance(new_route2)
                    if nd2 < best_dist:
                        best = new_route2
                        best_dist = nd2
                        improved = True
    return best


def intra_route_optimize(route):
    """ใช้ 2-opt แล้วตามด้วย or-opt แล้ว 2-opt อีกครั้ง"""
    r = two_opt(route)
    r = or_opt(r)
    r = two_opt(r)
    return r


def inter_route_relocate(routes):
    """ย้าย Customer จาก route หนึ่งไปอีก route (ถ้าทำให้ดีขึ้น)"""
    improved = True
    best_routes = [r[:] for r in routes]
    while improved:
        improved = False
        for r1_idx in range(len(best_routes)):
            for r2_idx in range(len(best_routes)):
                if r1_idx == r2_idx:
                    continue
                r1 = best_routes[r1_idx]
                r2 = best_routes[r2_idx]
                for i in range(1, len(r1) - 1):
                    node = r1[i]
                    if route_load(r2) + demands[node] > capacity:
                        continue
                    new_r1 = r1[:i] + r1[i+1:]
                    d_old = route_distance(r1) + route_distance(r2)
                    d_r1_new = route_distance(new_r1)
                    best_delta = 0
                    best_j = -1
                    for j in range(1, len(r2)):
                        new_r2 = r2[:j] + [node] + r2[j:]
                        d_r2_new = route_distance(new_r2)
                        delta = (d_r1_new + d_r2_new) - d_old
                        if delta < best_delta:
                            best_delta = delta
                            best_j = j
                    if best_j >= 0:
                        best_routes[r1_idx] = new_r1
                        best_routes[r2_idx] = r2[:best_j] + [node] + r2[best_j:]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    # [NEW] ลบ route ที่ว่างเปล่า (มีแค่ depot-depot)
    best_routes = [r for r in best_routes if len(r) > 2]
    return best_routes


def inter_route_swap(routes):
    """สลับ Customer คู่ระหว่าง routes (ถ้าทำให้ดีขึ้น)"""
    improved = True
    best_routes = [r[:] for r in routes]
    while improved:
        improved = False
        for r1_idx in range(len(best_routes)):
            for r2_idx in range(r1_idx + 1, len(best_routes)):
                r1 = best_routes[r1_idx]
                r2 = best_routes[r2_idx]
                for i in range(1, len(r1) - 1):
                    for j in range(1, len(r2) - 1):
                        n1, n2 = r1[i], r2[j]
                        new_load1 = route_load(r1) - demands[n1] + demands[n2]
                        new_load2 = route_load(r2) - demands[n2] + demands[n1]
                        if new_load1 > capacity or new_load2 > capacity:
                            continue
                        new_r1 = r1[:i] + [n2] + r1[i+1:]
                        new_r2 = r2[:j] + [n1] + r2[j+1:]
                        d_old = route_distance(r1) + route_distance(r2)
                        d_new = route_distance(new_r1) + route_distance(new_r2)
                        if d_new < d_old:
                            best_routes[r1_idx] = new_r1
                            best_routes[r2_idx] = new_r2
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return best_routes


def cross_exchange(routes, max_seg=4):
    """Cross-exchange: สลับ segment ระหว่าง 2 routes (จำกัดความยาว segment)"""
    improved = True
    best_routes = [r[:] for r in routes]
    while improved:
        improved = False
        for r1_idx in range(len(best_routes)):
            for r2_idx in range(r1_idx + 1, len(best_routes)):
                r1 = best_routes[r1_idx]
                r2 = best_routes[r2_idx]
                # [IMPROVED] จำกัดความยาว segment เพื่อลดเวลา O(n^4) -> O(n^2 * max_seg^2)
                for i1 in range(1, len(r1) - 1):
                    for i2 in range(i1, min(i1 + max_seg, len(r1) - 1)):
                        for j1 in range(1, len(r2) - 1):
                            for j2 in range(j1, min(j1 + max_seg, len(r2) - 1)):
                                seg1 = r1[i1:i2+1]
                                seg2 = r2[j1:j2+1]
                                new_r1 = r1[:i1] + seg2 + r1[i2+1:]
                                new_r2 = r2[:j1] + seg1 + r2[j2+1:]
                                if route_load(new_r1) > capacity or route_load(new_r2) > capacity:
                                    continue
                                d_old = route_distance(r1) + route_distance(r2)
                                d_new = route_distance(new_r1) + route_distance(new_r2)
                                if d_new < d_old:
                                    best_routes[r1_idx] = new_r1
                                    best_routes[r2_idx] = new_r2
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return best_routes


def full_optimization(routes, iterations=5):
    """รวม local search ทั้งหมดเข้าด้วยกัน ทำซ้ำจนไม่ดีขึ้น"""
    best = [r[:] for r in routes]
    best_total = total_distance(best)

    for _ in range(iterations):
        best = [intra_route_optimize(r) for r in best]
        best = inter_route_relocate(best)
        best = inter_route_swap(best)
        best = cross_exchange(best)
        best = [intra_route_optimize(r) for r in best]

        new_total = total_distance(best)
        if new_total >= best_total:
            break
        best_total = new_total

    return best


# [NEW] Perturbation — สุ่มเปลี่ยน route เพื่อหลุดจาก local optimum
def perturb_routes(routes):
    """สุ่มเปลี่ยน routes ด้วยวิธีต่างๆ เพื่อสร้างจุดเริ่มต้นใหม่"""
    new_routes = [r[:] for r in routes]
    method = random.choice(["reverse_segment", "move_customer", "swap_between"])

    if method == "reverse_segment" and new_routes:
        # กลับ segment สุ่มใน route สุ่ม
        ri = random.randint(0, len(new_routes) - 1)
        r = new_routes[ri]
        if len(r) > 4:
            i = random.randint(1, len(r) - 3)
            j = random.randint(i + 1, len(r) - 2)
            r[i:j+1] = r[i:j+1][::-1]
            new_routes[ri] = r

    elif method == "move_customer" and len(new_routes) >= 2:
        # ย้าย customer สุ่มข้าม route
        r1_idx = random.randint(0, len(new_routes) - 1)
        r1 = new_routes[r1_idx]
        if len(r1) > 3:
            i = random.randint(1, len(r1) - 2)
            node = r1[i]
            r2_idx = random.randint(0, len(new_routes) - 1)
            while r2_idx == r1_idx:
                r2_idx = random.randint(0, len(new_routes) - 1)
            r2 = new_routes[r2_idx]
            if route_load(r2) + demands[node] <= capacity:
                new_routes[r1_idx] = r1[:i] + r1[i+1:]
                j = random.randint(1, len(r2) - 1)
                new_routes[r2_idx] = r2[:j] + [node] + r2[j:]

    elif method == "swap_between" and len(new_routes) >= 2:
        # สลับ customer สุ่มระหว่าง 2 routes
        r1_idx, r2_idx = random.sample(range(len(new_routes)), 2)
        r1, r2 = new_routes[r1_idx], new_routes[r2_idx]
        if len(r1) > 2 and len(r2) > 2:
            i = random.randint(1, len(r1) - 2)
            j = random.randint(1, len(r2) - 2)
            n1, n2 = r1[i], r2[j]
            new_load1 = route_load(r1) - demands[n1] + demands[n2]
            new_load2 = route_load(r2) - demands[n2] + demands[n1]
            if new_load1 <= capacity and new_load2 <= capacity:
                new_routes[r1_idx] = r1[:i] + [n2] + r1[i+1:]
                new_routes[r2_idx] = r2[:j] + [n1] + r2[j+1:]

    # ลบ route ว่าง
    new_routes = [r for r in new_routes if len(r) > 2]
    return new_routes


# ============================================================
#  9. Multi-Strategy Solver
# ============================================================

start_time = time.time()

print("\n" + "=" * 55)
print("  กำลังหา Route ที่ดีที่สุด...")
print("=" * 55)

overall_best_routes = None
overall_best_total = float('inf')
overall_best_method = ""

def update_best(routes, method_name):
    """อัปเดตผลลัพธ์ที่ดีที่สุด"""
    global overall_best_routes, overall_best_total, overall_best_method
    t = total_distance(routes)
    if t < overall_best_total:
        overall_best_total = t
        overall_best_routes = [r[:] for r in routes]
        overall_best_method = method_name
        return True
    return False

def time_ok():
    """ยังอยู่ภายในเวลาที่กำหนดหรือไม่"""
    return (time.time() - start_time) < TIME_LIMIT


# --- Strategy 1: Basic Nearest Neighbor ---
nn_routes, nn_remaining = nearest_neighbor(depot, customers, max_vehicles)
if not nn_remaining:
    nn_opt = full_optimization(nn_routes)
    print(f"  NN + Local Search             : {total_distance(nn_opt)}")
    update_best(nn_opt, "Nearest Neighbor + Local Search")

# --- Strategy 2: Multi-start NN (ลองเริ่มจากทุก Customer) ---
if time_ok():
    for seed in customers:
        if not time_ok():
            print(f"  Multi-start NN      : หมดเวลา (ทดสอบได้ {customers.index(seed)} seeds)")
            break
        seed_routes, seed_remaining = nearest_neighbor_seed(depot, customers, max_vehicles, seed)
        if not seed_remaining:
            seed_opt = full_optimization(seed_routes)
            update_best(seed_opt, f"NN seed={seed} + Local Search")

print(f"  Multi-start NN + LS (best)    : {overall_best_total}")

# --- Strategy 3: Clarke-Wright Savings ---
if time_ok():
    cw_routes, cw_remaining = clarke_wright_savings(depot, customers, max_vehicles)
    if not cw_remaining:
        cw_opt = full_optimization(cw_routes)
        print(f"  Clarke-Wright + Local Search  : {total_distance(cw_opt)}")
        update_best(cw_opt, "Clarke-Wright Savings + Local Search")

# --- Strategy 4: Sweep Algorithm (ลองหลายมุมเริ่มต้น) --- [NEW]
if time_ok():
    sweep_best = float('inf')
    n_angles = 36  # ลอง 36 มุม (ทุก 10 องศา)
    for k in range(n_angles):
        if not time_ok():
            break
        angle = (2 * math.pi * k) / n_angles
        sw_routes, sw_remaining = sweep_algorithm(depot, customers, max_vehicles, angle)
        if not sw_remaining:
            sw_opt = full_optimization(sw_routes)
            sw_total = total_distance(sw_opt)
            if sw_total < sweep_best:
                sweep_best = sw_total
            update_best(sw_opt, f"Sweep angle={int(math.degrees(angle))}° + Local Search")
    print(f"  Sweep 36 มุม + LS (best)      : {sweep_best if sweep_best < float('inf') else 'N/A'}")

# --- Strategy 5: Exhaustive Partition (สำหรับ instance เล็ก) ---
if time_ok() and len(customers) <= 22 and max_vehicles >= 2:
    n = len(customers)
    exhaustive_ok = (max_vehicles == 2) or (max_vehicles == 3 and n <= 18)
    if exhaustive_ok:
        print("  Exhaustive partition: กำลังค้นหา...")
    if max_vehicles == 2:
        # 2 รถ: ลองทุก bitmask
        for mask in range(1, 2**(n-1)):
            if not time_ok():
                break
            group1 = [customers[i] for i in range(n) if mask & (1 << i)]
            group2 = [customers[i] for i in range(n) if not (mask & (1 << i))]

            load1 = sum(demands[c] for c in group1)
            load2 = sum(demands[c] for c in group2)
            if load1 > capacity or load2 > capacity:
                continue

            r1 = two_opt([depot] + group1 + [depot])
            r2 = two_opt([depot] + group2 + [depot])
            update_best([r1, r2], "Exhaustive Partition (2V) + 2-opt")
    elif max_vehicles == 3 and n <= 18:
        # [NEW] 3 รถ: ลองแบ่ง 3 กลุ่ม (ใช้ 2 bitmask ซ้อน)
        for mask1 in range(1, 2**n):
            if not time_ok():
                break
            group1 = [customers[i] for i in range(n) if mask1 & (1 << i)]
            load1 = sum(demands[c] for c in group1)
            if load1 > capacity:
                continue
            rest = [customers[i] for i in range(n) if not (mask1 & (1 << i))]
            nr = len(rest)
            for mask2 in range(1, 2**(nr-1)):
                group2 = [rest[i] for i in range(nr) if mask2 & (1 << i)]
                group3 = [rest[i] for i in range(nr) if not (mask2 & (1 << i))]
                load2 = sum(demands[c] for c in group2)
                load3 = sum(demands[c] for c in group3)
                if load2 > capacity or load3 > capacity:
                    continue
                r1 = two_opt([depot] + group1 + [depot])
                r2 = two_opt([depot] + group2 + [depot])
                r3 = two_opt([depot] + group3 + [depot])
                update_best([r1, r2, r3], "Exhaustive Partition (3V) + 2-opt")
    else:
        print(f"  Exhaustive partition: ข้ามไป (รองรับรถ 2 คัน หรือ 3 คันที่ไม่เกิน 18 จุด · ไฟล์นี้มีรถ {max_vehicles} คัน {n} จุด)")
    if exhaustive_ok:
        print(f"  Exhaustive best     : {overall_best_total}")
elif len(customers) > 22:
    print(f"  Exhaustive partition: ข้ามไป (customers={len(customers)} > 22)")

# --- Strategy 6: Perturbation (Iterated Local Search) --- [NEW]
if time_ok() and overall_best_routes:
    print(f"  Perturbation        : กำลังทำ (สูงสุด {PERTURB_ITER} รอบ)...")
    perturb_improved = 0
    for pi in range(PERTURB_ITER):
        if not time_ok():
            break
        perturbed = perturb_routes(overall_best_routes)
        perturbed_opt = full_optimization(perturbed)
        if update_best(perturbed_opt, f"Perturbation iter={pi+1}"):
            perturb_improved += 1
    print(f"  Perturbation found  : {perturb_improved} improvements")

# Final polish
if overall_best_routes:
    overall_best_routes = full_optimization(overall_best_routes, iterations=10)
    overall_best_total = total_distance(overall_best_routes)

elapsed = time.time() - start_time
print(f"\n  ✅ วิธีที่ดีที่สุด: {overall_best_method}")
print(f"  ✅ Total Distance : {overall_best_total}")
print(f"  ⏱  เวลาที่ใช้     : {elapsed:.1f} วินาที")

if overall_best_routes is None:
    print("\n❌ ไม่สามารถหาเส้นทางที่ตอบสนองเงื่อนไข (Capacity/จำนวนรถ) ได้เลย")
    print("อาจเป็นไปได้ว่า Demand รวมสูงกว่า Capacity รวมของรถทั้งหมด, หรือบรรจุไม่ลง")
    sys.exit(1)

# ใช้ผลลัพธ์ที่ดีที่สุด
routes = overall_best_routes
route_distances = [route_distance(r) for r in routes]
route_loads = [route_load(r) for r in routes]

# ==================== 10. แสดงผลลัพธ์ ====================
pd.set_option("display.max_colwidth", None)

print("\n" + "=" * 55)
print("  ผลลัพธ์ที่ดีที่สุด")
print("=" * 55)

summary_data = []
for i, (route, dist) in enumerate(zip(routes, route_distances)):
    cust_in_route = [n for n in route if n != depot]
    r_demand = sum(demands[n] for n in cust_in_route)
    r_str = " -> ".join(str(n) for n in route)
    summary_data.append({
        "Vehicle": i + 1,
        "Route": r_str,
        "Customers": len(cust_in_route),
        "Demand": r_demand,
        "Cap Left": capacity - r_demand,
        "Distance": dist,
    })

df_result = pd.DataFrame(summary_data)
print(df_result.to_string(index=False))

total_dist = sum(route_distances)
print(f"\nTotal Distance = {total_dist}")

# แสดง Gap ถ้ามี Optimal ใน COMMENT
if comment_text:
    opt_match = re.search(r"Optimal value:\s*(\d+)", comment_text, re.IGNORECASE)
    if opt_match:
        opt_val = int(opt_match.group(1))
        gap = (total_dist - opt_val) / opt_val * 100
        print(f"Optimal Value  = {opt_val}")
        print(f"Gap            = {gap:.2f}%")

# ==================== 11. Trace (3 การเลือกแรกของ NN เดิม) ====================
nn_routes_raw, _ = nearest_neighbor(depot, customers, max_vehicles)
trace = []
route0 = nn_routes_raw[0] if nn_routes_raw else []
for step in range(min(3, len(route0) - 2)):
    current = route0[step]
    selected = route0[step + 1]
    trace.append({
        "Step": step + 1,
        "Current Node": current,
        "Selected Node": selected,
        "EUC_2D": dist_matrix[(current, selected)],
        "Selected Demand": demands[selected],
        "Load After Selection": sum(demands[route0[k]] for k in range(1, step + 2)),
    })

print("\n" + "=" * 55)
print("  Trace: 3 การเลือกแรกของ NN (รถคันที่ 1)")
print("=" * 55)

if trace:
    df_trace = pd.DataFrame(trace)
    print(df_trace.to_string(index=False))
else:
    print("  ไม่มีข้อมูล Trace")

# ==================== 12. Validator ====================
def validate_routes(
    routes, depot, customers, demands,
    capacity, max_vehicles, reported_distances, reported_loads
):
    """ตรวจสอบ Route โดยคำนวณใหม่จากข้อมูลต้นฉบับ"""
    results = []
    all_pass = True

    all_cust = []
    for route in routes:
        all_cust.extend([n for n in route if n != depot])

    # 1. ทุก Node มีอยู่จริง
    invalid = [n for route in routes for n in route if n not in node_coords]
    if invalid:
        results.append(("1. Nodes exist in data", "FAIL", f"ไม่พบ Node: {invalid}"))
        all_pass = False
    else:
        results.append(("1. Nodes exist in data", "PASS", "ทุก Node มีอยู่จริง"))

    # 2. Customer ครบ
    missing = sorted(set(customers) - set(all_cust))
    if missing:
        results.append(("2. All customers served", "FAIL", f"หายไป: {missing}"))
        all_pass = False
    else:
        results.append(("2. All customers served", "PASS", "ครบทุกคน"))

    # 3. ไม่มี Customer ซ้ำ
    counts = Counter(all_cust)
    dups = {n: c for n, c in counts.items() if c > 1}
    if dups:
        results.append(("3. No duplicate customers", "FAIL", f"ซ้ำ: {dups}"))
        all_pass = False
    else:
        results.append(("3. No duplicate customers", "PASS", "ไม่มีซ้ำ"))

    # 4. Demand ไม่เกิน Capacity
    cap_ok = True
    for i, route in enumerate(routes):
        rd = sum(demands[n] for n in route if n != depot)
        if rd > capacity:
            results.append((f"4. Capacity (V{i+1})", "FAIL", f"Demand {rd} > {capacity}"))
            cap_ok = False
            all_pass = False
    if cap_ok:
        results.append(("4. Capacity", "PASS", "ทุกรถไม่เกิน Capacity"))

    # 4b. Load ตรงกับที่รายงาน
    load_ok = True
    for i, route in enumerate(routes):
        recalc = sum(demands[n] for n in route if n != depot)
        if recalc != reported_loads[i]:
            results.append((
                f"4b. Load match (V{i+1})", "FAIL",
                f"Reported {reported_loads[i]} vs Recalculated {recalc}"
            ))
            load_ok = False
            all_pass = False
    if load_ok:
        results.append(("4b. Load match", "PASS", "Demand ตรงกันทุกคัน"))

    # 5. จำนวนรถ
    if len(routes) > max_vehicles:
        results.append(("5. Vehicle count", "FAIL", f"{len(routes)} > {max_vehicles}"))
        all_pass = False
    else:
        results.append(("5. Vehicle count", "PASS", f"{len(routes)} ≤ {max_vehicles}"))

    # 6-7. เริ่ม/จบ Depot
    bad_start = [i+1 for i, r in enumerate(routes) if r[0] != depot]
    if bad_start:
        results.append(("6. Start at depot", "FAIL", f"รถที่ไม่เริ่ม Depot: {bad_start}"))
        all_pass = False
    else:
        results.append(("6. Start at depot", "PASS", "ทุก Route เริ่มที่ Depot"))

    bad_end = [i+1 for i, r in enumerate(routes) if r[-1] != depot]
    if bad_end:
        results.append(("7. End at depot", "FAIL", f"รถที่ไม่จบ Depot: {bad_end}"))
        all_pass = False
    else:
        results.append(("7. End at depot", "PASS", "ทุก Route จบที่ Depot"))

    # 8. ไม่มี Depot กลาง Route
    mid_bad = [i+1 for i, r in enumerate(routes) if depot in r[1:-1]]
    if mid_bad:
        results.append(("8. No depot in middle", "FAIL", f"Depot แทรกกลางในรถ: {mid_bad}"))
        all_pass = False
    else:
        results.append(("8. No depot in middle", "PASS", "ไม่มี Depot แทรกกลาง"))

    # 9-10. Distance ตรงกัน
    recalc = [route_distance(r) for r in routes]
    dist_ok = True
    for i in range(len(routes)):
        if recalc[i] != reported_distances[i]:
            results.append((
                f"9-10. Distance (V{i+1})", "FAIL",
                f"รายงาน {reported_distances[i]} แต่คำนวณใหม่ได้ {recalc[i]}"
            ))
            dist_ok = False
            all_pass = False
    if dist_ok:
        results.append(("9-10. Distance match", "PASS", f"ตรงกันทุกคัน (Total = {sum(recalc)})"))

    return results, all_pass


print("\n" + "=" * 55)
print("  Validation")
print("=" * 55)

val_results, overall_pass = validate_routes(
    routes, depot, customers, demands,
    capacity, max_vehicles, route_distances, route_loads,
)

for name, status, detail in val_results:
    mark = "✅" if status == "PASS" else "❌"
    print(f"  {mark} {status}  {name}: {detail}")

print("\n" + "-" * 55)
if overall_pass:
    print("  OVERALL VALIDATION: PASS ✅")
else:
    print("  OVERALL VALIDATION: FAIL ❌")


