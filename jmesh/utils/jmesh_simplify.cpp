// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <queue>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

/// Error quadric that is used to minimize the squared distance of a point to
/// its neigbhouring triangle planes.
/// Cf. "Simplifying Surfaces with Color and Texture using Quadric Error
/// Metrics" by Garland and Heckbert.
template <typename T>
struct hash_eigen {
    std::size_t operator()(T const& matrix) const {
        size_t seed = 0;
        for (int i = 0; i < (int)matrix.size(); i++) {
            auto elem = *(matrix.data() + i);
            seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 +
                    (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

class Quadric {
public:
    Quadric() {
        A_.fill(0);
        b_.fill(0);
        c_ = 0;
    }

    Quadric(const Eigen::Vector4d& plane, double weight = 1) {
        Eigen::Vector3d n = plane.head<3>();
        A_ = weight * n * n.transpose();
        b_ = weight * plane(3) * n;
        c_ = weight * plane(3) * plane(3);
    }

    Quadric& operator+=(const Quadric& other) {
        A_ += other.A_;
        b_ += other.b_;
        c_ += other.c_;
        return *this;
    }

    Quadric operator+(const Quadric& other) const {
        Quadric res;
        res.A_ = A_ + other.A_;
        res.b_ = b_ + other.b_;
        res.c_ = c_ + other.c_;
        return res;
    }

    double Eval(const Eigen::Vector3d& v) const {
        Eigen::Vector3d Av = A_ * v;
        double q = v.dot(Av) + 2 * b_.dot(v) + c_;
        return q;
    }

    bool IsInvertible() const { return std::fabs(A_.determinant()) > 1e-4; }

    Eigen::Vector3d Minimum() const { return -A_.ldlt().solve(b_); }

public:
    /// A_ = n . n^T, where n is the plane normal
    Eigen::Matrix3d A_;
    /// b_ = d . n, where n is the plane normal and d the non-normal component
    /// of the plane parameters
    Eigen::Vector3d b_;
    /// c_ = d . d, where d the non-normal component pf the plane parameters
    double c_;
};

class TriangleMesh {
    public:
        std::vector<Eigen::Vector3d> vertices_;
        std::vector<Eigen::Vector3d> vertex_colors_;
        std::vector<Eigen::Vector3i> triangles_;
        std::vector<bool> triangles_deleted_;

    public:
        
        TriangleMesh() {}
        TriangleMesh(const std::vector<Eigen::Vector3d> &vertices,
            const std::vector<Eigen::Vector3i> triangles):vertices_(vertices),triangles_(triangles){
                
        }
        ~TriangleMesh() {}
        Eigen::Vector4d ComputeTrianglePlane(const Eigen::Vector3d &p0,
                                                        const Eigen::Vector3d &p1,
                                                        const Eigen::Vector3d &p2) {
            const Eigen::Vector3d e0 = p1 - p0;
            const Eigen::Vector3d e1 = p2 - p0;
            Eigen::Vector3d abc = e0.cross(e1);
            double norm = abc.norm();
            // if the three points are co-linear, return invalid plane
            if (norm == 0) {
                return Eigen::Vector4d(0, 0, 0, 0);
            }
            abc /= abc.norm();
            double d = -abc.dot(p0);
            return Eigen::Vector4d(abc(0), abc(1), abc(2), d);
        }

        Eigen::Vector4d GetTrianglePlane(size_t triangle_idx){
            const Eigen::Vector3i &triangle = triangles_[triangle_idx];
            const Eigen::Vector3d &vertex0 = vertices_[triangle(0)];
            const Eigen::Vector3d &vertex1 = vertices_[triangle(1)];
            const Eigen::Vector3d &vertex2 = vertices_[triangle(2)];
            return ComputeTrianglePlane(vertex0, vertex1, vertex2);
        }

        double ComputeTriangleArea(const Eigen::Vector3d &p0,
                                                const Eigen::Vector3d &p1,
                                                const Eigen::Vector3d &p2) {
            const Eigen::Vector3d x = p0 - p1;
            const Eigen::Vector3d y = p0 - p2;
            double area = 0.5 * x.cross(y).norm();
            return area;
        }
        

        double GetTriangleArea(size_t triangle_idx){
            const Eigen::Vector3i &triangle = triangles_[triangle_idx];
            const Eigen::Vector3d &vertex0 = vertices_[triangle(0)];
            const Eigen::Vector3d &vertex1 = vertices_[triangle(1)];
            const Eigen::Vector3d &vertex2 = vertices_[triangle(2)];
            return ComputeTriangleArea(vertex0, vertex1, vertex2);
        }
        
        static inline Eigen::Vector2i GetOrderedEdge(int vidx0, int vidx1) {
            return Eigen::Vector2i(std::min(vidx0, vidx1), std::max(vidx0, vidx1));
        }

        std::unordered_map<Eigen::Vector2i,
                   std::vector<int>,
                   hash_eigen<Eigen::Vector2i>>
        GetEdgeToTrianglesMap() const {
            std::unordered_map<Eigen::Vector2i, std::vector<int>,
                            hash_eigen<Eigen::Vector2i>>
                    trias_per_edge;
            auto AddEdge = [&](int vidx0, int vidx1, int tidx) {
                trias_per_edge[GetOrderedEdge(vidx0, vidx1)].push_back(tidx);
            };
            for (size_t tidx = 0; tidx < triangles_.size(); ++tidx) {
                const auto &triangle = triangles_[tidx];
                AddEdge(triangle(0), triangle(1), int(tidx));
                AddEdge(triangle(1), triangle(2), int(tidx));
                AddEdge(triangle(2), triangle(0), int(tidx));
            }
            return trias_per_edge;
        }
        
       std::vector<int> SimplifyQuadricDecimation(
                int target_number_of_triangles,
                double maximum_error = std::numeric_limits<double>::infinity(),
                double boundary_weight = 1.0){

            typedef std::tuple<double, int, int> CostEdge;

            auto mesh = std::make_shared<TriangleMesh>();
            mesh->vertices_ = vertices_;
            mesh->vertex_colors_ = vertex_colors_;
            mesh->triangles_ = triangles_;
            
            std::vector<int> vertices_map(vertices_.size(), -1);
            std::vector<bool> vertices_deleted(vertices_.size(), false);
            std::vector<bool> triangles_deleted(triangles_.size(), false);

            // Map vertices to triangles and compute triangle planes and areas
            std::vector<std::unordered_set<int>> vert_to_triangles(vertices_.size());
            std::vector<Eigen::Vector4d> triangle_planes(triangles_.size());
            std::vector<double> triangle_areas(triangles_.size());
            for (size_t tidx = 0; tidx < triangles_.size(); ++tidx) {
                vert_to_triangles[triangles_[tidx](0)].emplace(static_cast<int>(tidx));
                vert_to_triangles[triangles_[tidx](1)].emplace(static_cast<int>(tidx));
                vert_to_triangles[triangles_[tidx](2)].emplace(static_cast<int>(tidx));

                triangle_planes[tidx] = GetTrianglePlane(tidx);
                triangle_areas[tidx] = GetTriangleArea(tidx);
            }

            // Compute the error metric per vertex
            std::vector<Quadric> Qs(vertices_.size());
            for (size_t vidx = 0; vidx < vertices_.size(); ++vidx) {
                for (int tidx : vert_to_triangles[vidx]) {
                    Qs[vidx] += Quadric(triangle_planes[tidx], triangle_areas[tidx]);
                }
            }

            // For boundary edges add perpendicular plane quadric
            auto edge_triangle_count = GetEdgeToTrianglesMap();
            auto AddPerpPlaneQuadric = [&](int vidx0, int vidx1, int vidx2,
                                        double area) {
                int min = std::min(vidx0, vidx1);
                int max = std::max(vidx0, vidx1);
                Eigen::Vector2i edge(min, max);
                if (edge_triangle_count[edge].size() != 1) {
                    return;
                }
                const auto& vert0 = mesh->vertices_[vidx0];
                const auto& vert1 = mesh->vertices_[vidx1];
                const auto& vert2 = mesh->vertices_[vidx2];
                Eigen::Vector3d vert2p = (vert2 - vert0).cross(vert2 - vert1);
                Eigen::Vector4d plane = ComputeTrianglePlane(vert0, vert1, vert2p);
                Quadric quad(plane, area * boundary_weight);
                Qs[vidx0] += quad;
                Qs[vidx1] += quad;
            };
            for (size_t tidx = 0; tidx < triangles_.size(); ++tidx) {
                const auto& tria = triangles_[tidx];
                double area = triangle_areas[tidx];
                AddPerpPlaneQuadric(tria(0), tria(1), tria(2), area);
                AddPerpPlaneQuadric(tria(1), tria(2), tria(0), area);
                AddPerpPlaneQuadric(tria(2), tria(0), tria(1), area);
            }

            // Get valid edges and compute cost
            // Note: We could also select all vertex pairs as edges with dist < eps
            std::unordered_map<Eigen::Vector2i, Eigen::Vector3d,
                                hash_eigen<Eigen::Vector2i>>
                    vbars;
            std::unordered_map<Eigen::Vector2i, double,
                                hash_eigen<Eigen::Vector2i>>
                    costs;
            auto CostEdgeComp = [](const CostEdge& a, const CostEdge& b) {
                return std::get<0>(a) > std::get<0>(b);
            };
            std::priority_queue<CostEdge, std::vector<CostEdge>, decltype(CostEdgeComp)>
                    queue(CostEdgeComp);

            auto AddEdge = [&](int vidx0, int vidx1, bool update) {
                int min = std::min(vidx0, vidx1);
                int max = std::max(vidx0, vidx1);
                Eigen::Vector2i edge(min, max);
                if (update || vbars.count(edge) == 0) {
                    const Quadric& Q0 = Qs[min];
                    const Quadric& Q1 = Qs[max];
                    Quadric Qbar = Q0 + Q1;
                    double cost;
                    Eigen::Vector3d vbar;
                    if (Qbar.IsInvertible()) {
                        vbar = Qbar.Minimum();
                        cost = Qbar.Eval(vbar);
                    } else {
                        const Eigen::Vector3d& v0 = mesh->vertices_[vidx0];
                        const Eigen::Vector3d& v1 = mesh->vertices_[vidx1];
                        Eigen::Vector3d vmid = (v0 + v1) / 2;
                        double cost0 = Qbar.Eval(v0);
                        double cost1 = Qbar.Eval(v1);
                        double costmid = Qbar.Eval(vmid);
                        cost = std::min(cost0, std::min(cost1, costmid));
                        if (cost == costmid) {
                            vbar = vmid;
                        } else if (cost == cost0) {
                            vbar = v0;
                        } else {
                            vbar = v1;
                        }
                    }
                    vbars[edge] = vbar;
                    costs[edge] = cost;
                    queue.push(CostEdge(cost, min, max));
                }
            };

            // add all edges to priority queue
            for (const auto& triangle : triangles_) {
                AddEdge(triangle(0), triangle(1), false);
                AddEdge(triangle(1), triangle(2), false);
                AddEdge(triangle(2), triangle(0), false);
            }

            // perform incremental edge collapse
            bool has_vert_color = vertex_colors_.size()>0 && vertex_colors_.size() == vertices_.size(); //HasVertexColors();
            int n_triangles = int(triangles_.size());
            while (n_triangles > target_number_of_triangles && !queue.empty()) {
                // retrieve edge from queue
                double cost;
                int vidx0, vidx1;
                std::tie(cost, vidx0, vidx1) = queue.top();
                queue.pop();

                if (cost > maximum_error) {
                    break;
                }

                // test if the edge has been updated (reinserted into queue)
                Eigen::Vector2i edge(vidx0, vidx1);
                bool valid = !vertices_deleted[vidx0] && !vertices_deleted[vidx1] &&
                            cost == costs[edge];
                if (!valid) {
                    continue;
                }

                // avoid flip of triangle normal
                bool flipped = false;
                for (int tidx : vert_to_triangles[vidx1]) {
                    if (triangles_deleted[tidx]) {
                        continue;
                    }

                    const Eigen::Vector3i& tria = mesh->triangles_[tidx];
                    bool has_vidx0 =
                            vidx0 == tria(0) || vidx0 == tria(1) || vidx0 == tria(2);
                    bool has_vidx1 =
                            vidx1 == tria(0) || vidx1 == tria(1) || vidx1 == tria(2);
                    if (has_vidx0 && has_vidx1) {
                        continue;
                    }

                    Eigen::Vector3d vert0 = mesh->vertices_[tria(0)];
                    Eigen::Vector3d vert1 = mesh->vertices_[tria(1)];
                    Eigen::Vector3d vert2 = mesh->vertices_[tria(2)];
                    Eigen::Vector3d norm_before = (vert1 - vert0).cross(vert2 - vert0);
                    norm_before /= norm_before.norm();

                    if (vidx1 == tria(0)) {
                        vert0 = vbars[edge];
                    } else if (vidx1 == tria(1)) {
                        vert1 = vbars[edge];
                    } else if (vidx1 == tria(2)) {
                        vert2 = vbars[edge];
                    }

                    Eigen::Vector3d norm_after = (vert1 - vert0).cross(vert2 - vert0);
                    norm_after /= norm_after.norm();
                    if (norm_before.dot(norm_after) < 0) {
                        flipped = true;
                        break;
                    }
                }
                if (flipped) {
                    continue;
                }

                // Connect triangles from vidx1 to vidx0, or mark deleted
                for (int tidx : vert_to_triangles[vidx1]) {
                    if (triangles_deleted[tidx]) {
                        continue;
                    }

                    Eigen::Vector3i& tria = mesh->triangles_[tidx];
                    bool has_vidx0 =
                            vidx0 == tria(0) || vidx0 == tria(1) || vidx0 == tria(2);
                    bool has_vidx1 =
                            vidx1 == tria(0) || vidx1 == tria(1) || vidx1 == tria(2);

                    if (has_vidx0 && has_vidx1) {
                        triangles_deleted[tidx] = true;
                        n_triangles--;
                        continue;
                    }

                    if (vidx1 == tria(0)) {
                        tria(0) = vidx0;
                    } else if (vidx1 == tria(1)) {
                        tria(1) = vidx0;
                    } else if (vidx1 == tria(2)) {
                        tria(2) = vidx0;
                    }
                    vert_to_triangles[vidx0].insert(tidx);
                }

                // update vertex vidx0 to vbar
                mesh->vertices_[vidx0] = vbars[edge];
                Qs[vidx0] += Qs[vidx1];
                if (has_vert_color) {
                    mesh->vertex_colors_[vidx0] = 0.5 * (mesh->vertex_colors_[vidx0] +
                                                        mesh->vertex_colors_[vidx1]);
                }
                vertices_deleted[vidx1] = true;
                vertices_map[vidx1] = vidx0;

                // Update edge costs for all triangles connecting to vidx0
                for (const auto& tidx : vert_to_triangles[vidx0]) {
                    if (triangles_deleted[tidx]) {
                        continue;
                    }
                    const Eigen::Vector3i& tria = mesh->triangles_[tidx];
                    if (tria(0) == vidx0 || tria(1) == vidx0) {
                        AddEdge(tria(0), tria(1), true);
                    }
                    if (tria(1) == vidx0 || tria(2) == vidx0) {
                        AddEdge(tria(1), tria(2), true);
                    }
                    if (tria(2) == vidx0 || tria(0) == vidx0) {
                        AddEdge(tria(2), tria(0), true);
                    }
                }
            }

            // Apply changes to the triangle mesh
            int next_free = 0;
            std::unordered_map<int, int> vert_remapping;
            for (size_t idx = 0; idx < mesh->vertices_.size(); ++idx) {
                if (!vertices_deleted[idx]) {
                    vert_remapping[int(idx)] = next_free;
                    mesh->vertices_[next_free] = mesh->vertices_[idx];
                    if (has_vert_color) {
                        mesh->vertex_colors_[next_free] = mesh->vertex_colors_[idx];
                    }
                    next_free++;
                }
            }
            for(size_t idx=0;idx<vertices_map.size();++idx){
                if(vertices_map[idx]!=-1){
                    int next_idx = vertices_map[idx];
                    while(vertices_map[next_idx]!=-1){
                        next_idx = vertices_map[next_idx];
                    }
                    vertices_map[idx] = next_idx;
                }
            }
            for(size_t idx=0;idx<vertices_map.size();++idx){
                if(vertices_map[idx] == -1){
                    vertices_map[idx] = -vert_remapping[int(idx)]-1;
                }else{
                    vertices_map[idx] = vert_remapping[vertices_map[idx]];
                }
            }
            mesh->vertices_.resize(next_free);
            if (has_vert_color) {
                mesh->vertex_colors_.resize(next_free);
            }

            next_free = 0;
            for (size_t idx = 0; idx < mesh->triangles_.size(); ++idx) {
                if (!triangles_deleted[idx]) {
                    Eigen::Vector3i tria = mesh->triangles_[idx];
                    mesh->triangles_[next_free](0) = vert_remapping[tria(0)];
                    mesh->triangles_[next_free](1) = vert_remapping[tria(1)];
                    mesh->triangles_[next_free](2) = vert_remapping[tria(2)];
                    next_free++;
                }
            }
            mesh->triangles_.resize(next_free);

            this->vertices_ = mesh->vertices_;
            this->vertex_colors_ = mesh->vertex_colors_;
            this->triangles_ = mesh->triangles_;
            this->triangles_deleted_ = triangles_deleted;
            return vertices_map;
        }

};


PYBIND11_MODULE(jmesh_simplify, m) {
    pybind11::class_<TriangleMesh> trianglemesh(m, "TriangleMesh");
    trianglemesh
        .def(pybind11::init<const std::vector<Eigen::Vector3d> &,
                            const std::vector<Eigen::Vector3i> &>(),
                    "Create a triangle mesh from vertices and triangle indices",
                    pybind11::arg("vertices"), pybind11::arg("triangles"))
        .def(pybind11::init())
        .def("simplify_quadric_decimation",
                 &TriangleMesh::SimplifyQuadricDecimation,
                 "",
                 pybind11::arg("target_number_of_triangles"),
                 pybind11::arg("maximum_error") = std::numeric_limits<double>::infinity(),
                 pybind11::arg("boundary_weight") = 1.0)
        .def_readwrite("vertices", &TriangleMesh::vertices_,
                "``float64`` array of shape ``(num_vertices, 3)``, "
                "use ``numpy.asarray()`` to access data: Vertex "
                "coordinates.")
        .def_readwrite("triangles_deleted", &TriangleMesh::triangles_deleted_,
                "``float64`` array of shape ``(num_vertices, 3)``, "
                "use ``numpy.asarray()`` to access data: Vertex "
                "coordinates.")
        .def_readwrite("triangles", &TriangleMesh::triangles_,
                           "``int`` array of shape ``(num_triangles, 3)``, use "
                           "``numpy.asarray()`` to access data: List of "
                           "triangles denoted by the index of points forming "
                           "the triangle.")
        .def_readwrite(
                "vertex_colors", &TriangleMesh::vertex_colors_,
                "``float64`` array of shape ``(num_vertices, 3)``, "
                "range ``[0, 1]`` , use ``numpy.asarray()`` to access "
                "data: RGB colors of vertices.");
}

int main(){
    return 0;
}