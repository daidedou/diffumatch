import numpy as np
import scipy as sp
import scipy.linalg as spLA
import scipy.sparse.linalg as sla
import os
import logging
import copy
import potpourri3d as pp3d
from sklearn.cluster import KMeans#, DBSCAN, SpectralClustering
from scipy.spatial import cKDTree
import trimesh as tm


try:
    from vtk import *
    import vtk.util.numpy_support as v2n

    gotVTK = True
except ImportError:
    print('could not import VTK functions')
    gotVTK = False


# import kernelFunctions as kfun

class vtkFields:
    def __init__(self):
        self.scalars = []
        self.vectors = []
        self.normals = []
        self.tensors = []


# General surface class, fibers possible. The fiber is a vector field of norm 1, defined on each vertex.
# It must be an array of the same size (number of vertices)x3.

class Surface:

    # Fibers: a list of vectors, with i-th element corresponding to the value of the vector field at vertex i
    # Contains as object:
    # vertices : all the vertices
    # centers: the centers of each face
    # faces: faces along with the id of the faces
    # surfel: surface element of each face (area*normal)
    # List of methods:
    # read : from filename type, call readFILENAME and set all surface attributes
    # updateVertices: update the whole surface after a modification of the vertices
    # computeVertexArea and Normals
    # getEdges
    # LocalSignedDistance distance function in the neighborhood of a shape
    # toPolyData: convert the surface to a polydata vtk object
    # fromPolyDate: guess what
    # Simplify : simplify the meshes
    # flipfaces: invert the indices of faces from [a, b, c] to [a, c, b]
    # smooth: get a smoother surface
    # Isosurface: compute isosurface
    # edgeRecove: ensure that orientation is correct
    # remove isolated: if isolated vertice, remove it
    # laplacianMatrix: compute Laplacian Matrix of the surface graph
    # graphLaplacianMatrix: ???
    # laplacianSegmentation: segment the surface using laplacian properties
    # surfVolume: compute volume inscribed in the surface (+ inscribed infinitesimal volume for each face)
    # surfCenter: compute surface Center
    # surfMoments: compute surface second order moments
    # surfU: compute the informations for surface rigid alignement
    # surfEllipsoid: compute ellipsoid representing the surface
    # savebyu of vitk or vtk2: save the surface in a file
    # concatenate: concatenate to another surface

    def __init__(self, surf=None, filename=None, FV=None):
        if surf == None:
            if FV == None:
                if filename == None:
                    self.vertices = np.empty(0)
                    self.centers = np.empty(0)
                    self.faces = np.empty(0)
                    self.surfel = np.empty(0)
                else:
                    if type(filename) is list:
                        fvl = []
                        for name in filename:
                            fvl.append(Surface(filename=name))
                        self.concatenate(fvl)
                    else:
                        self.read(filename)
            else:
                self.vertices = np.copy(FV[1])
                self.faces = np.int_(FV[0])
                self.computeCentersAreas()

        else:
            self.vertices = np.copy(surf.vertices)
            self.faces = np.copy(surf.faces)
            self.surfel = np.copy(surf.surfel)
            self.centers = np.copy(surf.centers)
            self.computeCentersAreas()
        self.volume, self.vols = self.surfVolume()
        self.center = self.surfCenter()
        self.cotanLaplacian()

    def read(self, filename):
        (mainPart, ext) = os.path.splitext(filename)
        if ext == '.byu':
            self.readbyu(filename)
        elif ext == '.off':
            self.readOFF(filename)
        elif ext == '.vtk':
            self.readVTK(filename)
        elif ext == '.obj':
            self.readOBJ(filename)
        elif ext == '.ply':
            self.readPLY(filename)
        elif ext == '.tri' or ext == ".ntri":
            self.readTRI(filename)
        else:
            print('Unknown Surface Extension:', ext)
            self.vertices = np.empty(0)
            self.centers = np.empty(0)
            self.faces = np.empty(0)
            self.surfel = np.empty(0)

    # face centers and area weighted normal
    def computeCentersAreas(self):
        xDef1 = self.vertices[self.faces[:, 0], :]
        xDef2 = self.vertices[self.faces[:, 1], :]
        xDef3 = self.vertices[self.faces[:, 2], :]
        self.centers = (xDef1 + xDef2 + xDef3) / 3
        self.surfel = np.cross(xDef2 - xDef1, xDef3 - xDef1)

    # modify vertices without toplogical change
    def updateVertices(self, x0):
        self.vertices = np.copy(x0)
        xDef1 = self.vertices[self.faces[:, 0], :]
        xDef2 = self.vertices[self.faces[:, 1], :]
        xDef3 = self.vertices[self.faces[:, 2], :]
        self.centers = (xDef1 + xDef2 + xDef3) / 3
        self.surfel = np.cross(xDef2 - xDef1, xDef3 - xDef1)
        self.volume, self.vols = self.surfVolume()

    def computeVertexArea(self):
        # compute areas of faces and vertices
        V = self.vertices
        F = self.faces
        nv = V.shape[0]
        nf = F.shape[0]
        AF = np.zeros([nf, 1])
        AV = np.zeros([nv, 1])
        for k in range(nf):
            # determining if face is obtuse
            x12 = V[F[k, 1], :] - V[F[k, 0], :]
            x13 = V[F[k, 2], :] - V[F[k, 0], :]
            n12 = np.sqrt((x12 ** 2).sum())
            n13 = np.sqrt((x13 ** 2).sum())
            c1 = (x12 * x13).sum() / (n12 * n13)
            x23 = V[F[k, 2], :] - V[F[k, 1], :]
            n23 = np.sqrt((x23 ** 2).sum())
            # n23 = norm(x23) ;
            c2 = -(x12 * x23).sum() / (n12 * n23)
            c3 = (x13 * x23).sum() / (n13 * n23)
            AF[k] = np.sqrt((np.cross(x12, x13) ** 2).sum()) / 2
            if (c1 < 0):
                # face obtuse at vertex 1
                AV[F[k, 0]] += AF[k] / 2
                AV[F[k, 1]] += AF[k] / 4
                AV[F[k, 2]] += AF[k] / 4
            elif (c2 < 0):
                # face obuse at vertex 2
                AV[F[k, 0]] += AF[k] / 4
                AV[F[k, 1]] += AF[k] / 2
                AV[F[k, 2]] += AF[k] / 4
            elif (c3 < 0):
                # face obtuse at vertex 3
                AV[F[k, 0]] += AF[k] / 4
                AV[F[k, 1]] += AF[k] / 4
                AV[F[k, 2]] += AF[k] / 2
            else:
                # non obtuse face
                cot1 = c1 / np.sqrt(1 - c1 ** 2)
                cot2 = c2 / np.sqrt(1 - c2 ** 2)
                cot3 = c3 / np.sqrt(1 - c3 ** 2)
                AV[F[k, 0]] += ((x12 ** 2).sum() * cot3 + (x13 ** 2).sum() * cot2) / 8
                AV[F[k, 1]] += ((x12 ** 2).sum() * cot3 + (x23 ** 2).sum() * cot1) / 8
                AV[F[k, 2]] += ((x13 ** 2).sum() * cot2 + (x23 ** 2).sum() * cot1) / 8

        for k in range(nv):
            if (np.fabs(AV[k]) < 1e-10):
                print('Warning: vertex ', k, 'has no face; use removeIsolated')
        # print('sum check area:', AF.sum(), AV.sum()
        return AV, AF

    def computeVertexNormals(self):
        self.computeCentersAreas()
        normals = np.zeros(self.vertices.shape)
        F = self.faces
        for k in range(F.shape[0]):
            normals[F[k, 0]] += self.surfel[k]
            normals[F[k, 1]] += self.surfel[k]
            normals[F[k, 2]] += self.surfel[k]
        af = np.sqrt((normals ** 2).sum(axis=1))
        # logging.info('min area = %.4f'%(af.min()))
        normals /= af.reshape([self.vertices.shape[0], 1])

        return normals

    # Computes edges from vertices/faces
    def getEdges(self):
        self.edges = []

        for k in range(self.faces.shape[0]):
            for kj in (0, 1, 2):
                u = [self.faces[k, kj], self.faces[k, (kj + 1) % 3]]

                if (u not in self.edges) & (u.reverse() not in self.edges):
                    self.edges.append(u)

        self.edgeFaces = []

        for u in self.edges:
            self.edgeFaces.append([])

        for k in range(self.faces.shape[0]):
            for kj in (0, 1, 2):
                u = [self.faces[k, kj], self.faces[k, (kj + 1) % 3]]

                if u in self.edges:
                    kk = self.edges.index(u)
                else:
                    u.reverse()
                    kk = self.edges.index(u)

                self.edgeFaces[kk].append(k)

        self.edges = np.int_(np.array(self.edges))
        self.bdry = np.int_(np.zeros(self.edges.shape[0]))

        for k in range(self.edges.shape[0]):
            if len(self.edgeFaces[k]) < 2:
                self.bdry[k] = 1

    # computes the signed distance function in a small neighborhood of a shape
    def LocalSignedDistance(self, data, value):
        d2 = 2 * np.array(data >= value) - 1
        c2 = np.cumsum(d2, axis=0)
        for j in range(2):
            c2 = np.cumsum(c2, axis=j + 1)
        (n0, n1, n2) = c2.shape

        rad = 3
        diam = 2 * rad + 1
        (x, y, z) = np.mgrid[-rad:rad + 1, -rad:rad + 1, -rad:rad + 1]
        cube = (x ** 2 + y ** 2 + z ** 2)
        maxval = (diam) ** 3
        s = 3.0 * rad ** 2
        res = d2 * s
        u = maxval * np.ones(c2.shape)
        u[rad + 1:n0 - rad, rad + 1:n1 - rad, rad + 1:n2 - rad] = (c2[diam:n0, diam:n1, diam:n2]
                                                                   - c2[0:n0 - diam, diam:n1, diam:n2] - c2[diam:n0,
                                                                                                         0:n1 - diam,
                                                                                                         diam:n2] - c2[
                                                                                                                    diam:n0,
                                                                                                                    diam:n1,
                                                                                                                    0:n2 - diam]
                                                                   + c2[0:n0 - diam, 0:n1 - diam, diam:n2] + c2[diam:n0,
                                                                                                             0:n1 - diam,
                                                                                                             0:n2 - diam] + c2[
                                                                                                                            0:n0 - diam,
                                                                                                                            diam:n1,
                                                                                                                            0:n2 - diam]
                                                                   - c2[0:n0 - diam, 0:n1 - diam, 0:n2 - diam])

        I = np.nonzero(np.fabs(u) < maxval)
        # print(len(I[0]))

        for k in range(len(I[0])):
            p = np.array((I[0][k], I[1][k], I[2][k]))
            bmin = p - rad
            bmax = p + rad + 1
            # print(p, bmin, bmax)
            if (d2[p[0], p[1], p[2]] > 0):
                # print(u[p[0],p[1], p[2]])
                # print(d2[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]].sum())
                res[p[0], p[1], p[2]] = min(
                    cube[np.nonzero(d2[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]] < 0)]) - .25
            else:
                res[p[0], p[1], p[2]] = - min(
                    cube[np.nonzero(d2[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]] > 0)]) - .25

        return res

    def toPolyData(self):
        if gotVTK:
            points = vtkPoints()
            for k in range(self.vertices.shape[0]):
                points.InsertNextPoint(self.vertices[k, 0], self.vertices[k, 1], self.vertices[k, 2])
            polys = vtkCellArray()
            for k in range(self.faces.shape[0]):
                polys.InsertNextCell(3)
                for kk in range(3):
                    polys.InsertCellPoint(self.faces[k, kk])
            polydata = vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(polys)
            return polydata
        else:
            raise Exception('Cannot run toPolyData without VTK')

    def fromPolyData(self, g, scales=[1., 1., 1.]):
        npoints = int(g.GetNumberOfPoints())
        nfaces = int(g.GetNumberOfPolys())
        logging.info('Dimensions: %d %d %d' % (npoints, nfaces, g.GetNumberOfCells()))
        V = np.zeros([npoints, 3])
        for kk in range(npoints):
            V[kk, :] = np.array(g.GetPoint(kk))
            # print(kk, V[kk])
            # print(kk, np.array(g.GetPoint(kk)))
        F = np.zeros([nfaces, 3])
        gf = 0
        for kk in range(g.GetNumberOfCells()):
            c = g.GetCell(kk)
            if (c.GetNumberOfPoints() == 3):
                for ll in range(3):
                    F[gf, ll] = c.GetPointId(ll)
                    # print(kk, gf, F[gf])
                gf += 1

                # self.vertices = np.multiply(data.shape-V-1, scales)
        self.vertices = np.multiply(V, scales)
        self.faces = np.int_(F[0:gf, :])
        self.computeCentersAreas()

    def Simplify(self, target=1000.0):
        if gotVTK:
            polydata = self.toPolyData()
            dc = vtkQuadricDecimation()
            red = 1 - min(np.float(target) / polydata.GetNumberOfPoints(), 1)
            dc.SetTargetReduction(red)
            dc.SetInput(polydata)
            dc.Update()
            g = dc.GetOutput()
            self.fromPolyData(g)
            z = self.surfVolume()
            if (z > 0):
                self.flipFaces()
                print('flipping volume', z, self.surfVolume())
        else:
            raise Exception('Cannot run Simplify without VTK')

    def flipFaces(self):
        self.faces = self.faces[:, [0, 2, 1]]
        self.computeCentersAreas()

    def smooth(self, n=30, smooth=0.1):
        if gotVTK:
            g = self.toPolyData()
            print(g)
            smoother = vtkWindowedSincPolyDataFilter()
            smoother.SetInput(g)
            smoother.SetNumberOfIterations(n)
            smoother.SetPassBand(smooth)
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            smoother.GenerateErrorScalarsOn()
            # smoother.GenerateErrorVectorsOn()
            smoother.Update()
            g = smoother.GetOutput()
            self.fromPolyData(g)
        else:
            raise Exception('Cannot run smooth without VTK')

    # Computes isosurfaces using vtk
    def Isosurface(self, data, value=0.5, target=1000.0, scales=[1., 1., 1.], smooth=0.1, fill_holes=1.):
        if gotVTK:
            # data = self.LocalSignedDistance(data0, value)
            if isinstance(data, vtkImageData):
                img = data
            else:
                img = vtkImageData()
                img.SetDimensions(data.shape)
                img.SetOrigin(0, 0, 0)
                if vtkVersion.GetVTKMajorVersion() >= 6:
                    img.AllocateScalars(VTK_FLOAT, 1)
                else:
                    img.SetNumberOfScalarComponents(1)
                v = vtkDoubleArray()
                v.SetNumberOfValues(data.size)
                v.SetNumberOfComponents(1)
                for ii, tmp in enumerate(np.ravel(data, order='F')):
                    v.SetValue(ii, tmp)
                    img.GetPointData().SetScalars(v)

            cf = vtkContourFilter()
            if vtkVersion.GetVTKMajorVersion() >= 6:
                cf.SetInputData(img)
            else:
                cf.SetInput(img)
            cf.SetValue(0, value)
            cf.SetNumberOfContours(1)
            cf.Update()
            # print(cf
            connectivity = vtkPolyDataConnectivityFilter()
            connectivity.ScalarConnectivityOff()
            connectivity.SetExtractionModeToLargestRegion()
            if vtkVersion.GetVTKMajorVersion() >= 6:
                connectivity.SetInputData(cf.GetOutput())
            else:
                connectivity.SetInput(cf.GetOutput())
            connectivity.Update()
            g = connectivity.GetOutput()

            if smooth > 0:
                smoother = vtkWindowedSincPolyDataFilter()
                if vtkVersion.GetVTKMajorVersion() >= 6:
                    smoother.SetInputData(g)
                else:
                    smoother.SetInput(g)
                #     else:
                # smoother.SetInputConnection(contour.GetOutputPort())
                smoother.SetNumberOfIterations(30)
                # this has little effect on the error!
                # smoother.BoundarySmoothingOff()
                # smoother.FeatureEdgeSmoothingOff()
                # smoother.SetFeatureAngle(120.0)
                smoother.SetPassBand(smooth)  # this increases the error a lot!
                smoother.NonManifoldSmoothingOn()
                # smoother.NormalizeCoordinatesOn()
                # smoother.GenerateErrorScalarsOn()
                # smoother.GenerateErrorVectorsOn()
                smoother.Update()
                g = smoother.GetOutput()

            # dc = vtkDecimatePro()
            red = 1 - min(np.float(target) / g.GetNumberOfPoints(), 1)
            # print('Reduction: ', red)
            dc = vtkQuadricDecimation()
            dc.SetTargetReduction(red)
            # dc.AttributeErrorMetricOn()
            # dc.SetDegree(10)
            # dc.SetSplitting(0)
            if vtkVersion.GetVTKMajorVersion() >= 6:
                dc.SetInputData(g)
            else:
                dc.SetInput(g)
                # dc.SetInput(g)
            # print(dc)
            dc.Update()
            g = dc.GetOutput()
            # print('points:', g.GetNumberOfPoints())
            cp = vtkCleanPolyData()
            if vtkVersion.GetVTKMajorVersion() >= 6:
                cp.SetInputData(dc.GetOutput())
            else:
                cp.SetInput(dc.GetOutput())
                #        cp.SetInput(dc.GetOutput())
            # cp.SetPointMerging(1)
            cp.ConvertPolysToLinesOn()
            cp.SetAbsoluteTolerance(1e-5)
            cp.Update()
            g = cp.GetOutput()
            self.fromPolyData(g, scales)
            z = self.surfVolume()
            if (z > 0):
                self.flipFaces()
                # print('flipping volume', z, self.surfVolume())
                logging.info('flipping volume %.2f %.2f' % (z, self.surfVolume()))

            # print(g)
            # npoints = int(g.GetNumberOfPoints())
            # nfaces = int(g.GetNumberOfPolys())
            # print('Dimensions:', npoints, nfaces, g.GetNumberOfCells())
            # V = np.zeros([npoints, 3])
            # for kk in range(npoints):
            #     V[kk, :] = np.array(g.GetPoint(kk))
            #     #print(kk, V[kk])
            #     #print(kk, np.array(g.GetPoint(kk)))
            # F = np.zeros([nfaces, 3])
            # gf = 0
            # for kk in range(g.GetNumberOfCells()):
            #     c = g.GetCell(kk)
            #     if(c.GetNumberOfPoints() == 3):
            #         for ll in range(3):
            #             F[gf,ll] = c.GetPointId(ll)
            #             #print(kk, gf, F[gf])
            #         gf += 1

            #         #self.vertices = np.multiply(data.shape-V-1, scales)
            # self.vertices = np.multiply(V, scales)
            # self.faces = np.int_(F[0:gf, :])
            # self.computeCentersAreas()
        else:
            raise Exception('Cannot run Isosurface without VTK')

    # Ensures that orientation is correct
    def edgeRecover(self):
        v = self.vertices
        f = self.faces
        nv = v.shape[0]
        nf = f.shape[0]
        # faces containing each oriented edge
        edg0 = np.int_(np.zeros((nv, nv)))
        # number of edges between each vertex
        edg = np.int_(np.zeros((nv, nv)))
        # contiguous faces
        edgF = np.int_(np.zeros((nf, nf)))
        for (kf, c) in enumerate(f):
            if (edg0[c[0], c[1]] > 0):
                edg0[c[1], c[0]] = kf + 1
            else:
                edg0[c[0], c[1]] = kf + 1

            if (edg0[c[1], c[2]] > 0):
                edg0[c[2], c[1]] = kf + 1
            else:
                edg0[c[1], c[2]] = kf + 1

            if (edg0[c[2], c[0]] > 0):
                edg0[c[0], c[2]] = kf + 1
            else:
                edg0[c[2], c[0]] = kf + 1

            edg[c[0], c[1]] += 1
            edg[c[1], c[2]] += 1
            edg[c[2], c[0]] += 1

        for kv in range(nv):
            I2 = np.nonzero(edg0[kv, :])
            for kkv in I2[0].tolist():
                edgF[edg0[kkv, kv] - 1, edg0[kv, kkv] - 1] = kv + 1

        isOriented = np.int_(np.zeros(f.shape[0]))
        isActive = np.int_(np.zeros(f.shape[0]))
        I = np.nonzero(np.squeeze(edgF[0, :]))
        # list of faces to be oriented
        activeList = [0] + I[0].tolist()
        lastOriented = 0
        isOriented[0] = True
        for k in activeList:
            isActive[k] = True

        while lastOriented < len(activeList) - 1:
            i = activeList[lastOriented]
            j = activeList[lastOriented + 1]
            I = np.nonzero(edgF[j, :])
            foundOne = False
            for kk in I[0].tolist():
                if (foundOne == False) & (isOriented[kk]):
                    foundOne = True
                    u1 = edgF[j, kk] - 1
                    u2 = edgF[kk, j] - 1
                    if not ((edg[u1, u2] == 1) & (edg[u2, u1] == 1)):
                        # reorient face j
                        edg[f[j, 0], f[j, 1]] -= 1
                        edg[f[j, 1], f[j, 2]] -= 1
                        edg[f[j, 2], f[j, 0]] -= 1
                        a = f[j, 1]
                        f[j, 1] = f[j, 2]
                        f[j, 2] = a
                        edg[f[j, 0], f[j, 1]] += 1
                        edg[f[j, 1], f[j, 2]] += 1
                        edg[f[j, 2], f[j, 0]] += 1
                elif (not isActive[kk]):
                    activeList.append(kk)
                    isActive[kk] = True
            if foundOne:
                lastOriented = lastOriented + 1
                isOriented[j] = True
                # print('oriented face', j, lastOriented,  'out of',  nf,  ';  total active', len(activeList))
            else:
                print('Unable to orient face', j)
                return
        self.vertices = v;
        self.faces = f;

        z, _ = self.surfVolume()
        if (z > 0):
            self.flipFaces()

    def removeIsolated(self):
        N = self.vertices.shape[0]
        inFace = np.int_(np.zeros(N))
        for k in range(3):
            inFace[self.faces[:, k]] = 1
        J = np.nonzero(inFace)
        self.vertices = self.vertices[J[0], :]
        logging.info('Found %d isolated vertices' % (J[0].shape[0]))
        Q = -np.ones(N)
        for k, j in enumerate(J[0]):
            Q[j] = k
        self.faces = np.int_(Q[self.faces])

    def laplacianMatrix(self):
        F = self.faces
        V = self.vertices;
        nf = F.shape[0]
        nv = V.shape[0]

        AV, AF = self.computeVertexArea()

        # compute edges and detect boundary
        # edm = sp.lil_matrix((nv,nv))
        edm = -np.ones([nv, nv]).astype(np.int32)
        E = np.zeros([3 * nf, 2]).astype(np.int32)
        j = 0
        for k in range(nf):
            if (edm[F[k, 0], F[k, 1]] == -1):
                edm[F[k, 0], F[k, 1]] = j
                edm[F[k, 1], F[k, 0]] = j
                E[j, :] = [F[k, 0], F[k, 1]]
                j = j + 1
            if (edm[F[k, 1], F[k, 2]] == -1):
                edm[F[k, 1], F[k, 2]] = j
                edm[F[k, 2], F[k, 1]] = j
                E[j, :] = [F[k, 1], F[k, 2]]
                j = j + 1
            if (edm[F[k, 0], F[k, 2]] == -1):
                edm[F[k, 2], F[k, 0]] = j
                edm[F[k, 0], F[k, 2]] = j
                E[j, :] = [F[k, 2], F[k, 0]]
                j = j + 1
        E = E[0:j, :]

        edgeFace = np.zeros([j, nf])
        ne = j
        # print(E)
        for k in range(nf):
            edgeFace[edm[F[k, 0], F[k, 1]], k] = 1
            edgeFace[edm[F[k, 1], F[k, 2]], k] = 1
            edgeFace[edm[F[k, 2], F[k, 0]], k] = 1

        bEdge = np.zeros([ne, 1])
        bVert = np.zeros([nv, 1])
        edgeAngles = np.zeros([ne, 2])
        for k in range(ne):
            I = np.flatnonzero(edgeFace[k, :])
            # print('I=', I, F[I, :], E.shape)
            # print('E[k, :]=', k, E[k, :])
            # print(k, edgeFace[k, :])
            for u in range(len(I)):
                f = I[u]
                i1l = np.flatnonzero(F[f, :] == E[k, 0])
                i2l = np.flatnonzero(F[f, :] == E[k, 1])
                # print(f, F[f, :])
                # print(i1l, i2l)
                i1 = i1l[0]
                i2 = i2l[0]
                s = i1 + i2
                if s == 1:
                    i3 = 2
                elif s == 2:
                    i3 = 1
                elif s == 3:
                    i3 = 0
                x1 = V[F[f, i1], :] - V[F[f, i3], :]
                x2 = V[F[f, i2], :] - V[F[f, i3], :]
                a = (np.cross(x1, x2) * np.cross(V[F[f, 1], :] - V[F[f, 0], :], V[F[f, 2], :] - V[F[f, 0], :])).sum()
                b = (x1 * x2).sum()
                if (a > 0):
                    edgeAngles[k, u] = b / np.sqrt(a)
                else:
                    edgeAngles[k, u] = b / np.sqrt(-a)
            if (len(I) == 1):
                # boundary edge
                bEdge[k] = 1
                bVert[E[k, 0]] = 1
                bVert[E[k, 1]] = 1
                edgeAngles[k, 1] = 0

                # Compute Laplacian matrix
        L = np.zeros([nv, nv])

        for k in range(ne):
            L[E[k, 0], E[k, 1]] = (edgeAngles[k, 0] + edgeAngles[k, 1]) / 2
            L[E[k, 1], E[k, 0]] = L[E[k, 0], E[k, 1]]

        for k in range(nv):
            L[k, k] = - L[k, :].sum()

        A = np.zeros([nv, nv])
        for k in range(nv):
            A[k, k] = AV[k]

        return L, A

    def graphLaplacianMatrix(self):
        F = self.faces
        V = self.vertices
        nf = F.shape[0]
        nv = V.shape[0]

        # compute edges and detect boundary
        # edm = sp.lil_matrix((nv,nv))
        edm = -np.ones([nv, nv])
        E = np.zeros([3 * nf, 2])
        j = 0
        for k in range(nf):
            if (edm[F[k, 0], F[k, 1]] == -1):
                edm[F[k, 0], F[k, 1]] = j
                edm[F[k, 1], F[k, 0]] = j
                E[j, :] = [F[k, 0], F[k, 1]]
                j = j + 1
            if (edm[F[k, 1], F[k, 2]] == -1):
                edm[F[k, 1], F[k, 2]] = j
                edm[F[k, 2], F[k, 1]] = j
                E[j, :] = [F[k, 1], F[k, 2]]
                j = j + 1
            if (edm[F[k, 0], F[k, 2]] == -1):
                edm[F[k, 2], F[k, 0]] = j
                edm[F[k, 0], F[k, 2]] = j
                E[j, :] = [F[k, 2], F[k, 0]]
                j = j + 1
        E = E[0:j, :]

        edgeFace = np.zeros([j, nf])
        ne = j
        # print(E)

        # Compute Laplacian matrix
        L = np.zeros([nv, nv])

        for k in range(ne):
            L[E[k, 0], E[k, 1]] = 1
            L[E[k, 1], E[k, 0]] = 1

        for k in range(nv):
            L[k, k] = - L[k, :].sum()

        return L

    def cotanLaplacian(self, eps=1e-8):
        L = pp3d.cotan_laplacian(self.vertices, self.faces, denom_eps=1e-10)
        massvec_np = pp3d.vertex_areas(self.vertices, self.faces)
        massvec_np += eps * np.mean(massvec_np)

        if (np.isnan(L.data).any()):
            raise RuntimeError("NaN Laplace matrix")
        if (np.isnan(massvec_np).any()):
            raise RuntimeError("NaN mass matrix")
        self.L = L 
        self.massvec_np = massvec_np
        return L, massvec_np

    def lapEigen(self, n_eig=10, eps=1e-8):
        L_eigsh = (self.L + sp.sparse.identity(self.L.shape[0]) * eps).tocsc()
        massvec_eigsh = self.massvec_np
        Mmat = sp.sparse.diags(massvec_eigsh)
        eigs_sigma = eps#-0.01
        evals, evecs = sla.eigsh(L_eigsh, k=n_eig, M=Mmat, sigma=eigs_sigma)
        return evals, evecs

    def laplacianSegmentation(self, k, verbose=False):
        # (L, AA) = self.laplacianMatrix()
        # # print((L.shape[0]-k-1, L.shape[0]-2))
        # (D, y) = spLA.eigh(L, AA, eigvals=(L.shape[0] - k, L.shape[0] - 1))
        evals, evecs = self.lapEigen(k*2)
        y = evecs[:, 1:] * np.exp(-2*0.1 * evals[1:]) #**2
        # N = y.shape[0]
        # d = y.shape[1]
        # I = np.argsort(y.sum(axis=1))
        # I0 = np.floor((N - 1) * sp.linspace(0, 1, num=k)).astype(int)
        # # print(y.shape, L.shape, N, k, d)
        # C = y[I0, :].copy()
        #
        # eps = 1e-20
        # Cold = C.copy()
        # u = ((C.reshape([k, 1, d]) - y.reshape([1, N, d])) ** 2).sum(axis=2)
        # T = u.min(axis=0).sum() / (N)
        # # print(T)
        # j = 0
        # while j < 5000:
        #     u0 = u - u.min(axis=0).reshape([1, N])
        #     w = np.exp(-u0 / T);
        #     w = w / (eps + w.sum(axis=0).reshape([1, N]))
        #     # print(w.min(), w.max())
        #     cost = (u * w).sum() + T * (w * np.log(w + eps)).sum()
        #     C = np.dot(w, y) / (eps + w.sum(axis=1).reshape([k, 1]))
        #     # print(j, 'cost0 ', cost)
        #
        #     u = ((C.reshape([k, 1, d]) - y.reshape([1, N, d])) ** 2).sum(axis=2)
        #     cost = (u * w).sum() + T * (w * np.log(w + eps)).sum()
        #     err = np.sqrt(((C - Cold) ** 2).sum(axis=1)).sum()
        #     # print(j, 'cost ', cost, err, T)
        #     if (j > 100) & (err < 1e-4):
        #         break
        #     j = j + 1
        #     Cold = C.copy()
        #     T = T * 0.99
        #
        #     # print(k, d, C.shape)
        # dst = ((C.reshape([k, 1, d]) - y.reshape([1, N, d])) ** 2).sum(axis=2)
        # md = dst.min(axis=0)
        # idx = np.zeros(N).astype(int)
        # for j in range(N):
        #     I = np.flatnonzero(dst[:, j] < md[j] + 1e-10)
        #     idx[j] = I[0]
        # I = -np.ones(k).astype(int)
        # kk = 0
        # for j in range(k):
        #     if True in (idx == j):
        #         I[j] = kk
        #         kk += 1
        # idx = I[idx]
        # if idx.max() < (k - 1):
        #     print('Warning: kmeans convergence with %d clusters instead of %d' % (idx.max(), k))
        #     # ml = w.sum(axis=1)/N
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(y)
        # kmeans = DBSCAN().fit(y)
        idx = kmeans.labels_
        nc = idx.max() + 1
        C = np.zeros([nc, self.vertices.shape[1]])
        a, foo = self.computeVertexArea()
        for k in range(nc):
            I = np.flatnonzero(idx == k)
            nI = len(I)
            # print(a.shape, nI)
            aI = a[I]
            ak = aI.sum()
            C[k, :] = (self.vertices[I, :] * aI).sum(axis=0) / ak;
        mean_eigen = (y*a).sum(axis=0)/a
        tree = cKDTree(y)
        _, indices = tree.query(mean_eigen, k=1)
        index_center = indices[0]
        _, indices = tree.query(C, k=1)
        index_C = indices[0]

        mesh = tm.Trimesh(vertices=self.vertices, faces=self.faces)
        nv = self.computeVertexNormals()
        rori = self.vertices[index] + 0.001 * (-nv[index, :])
        rdir = -nv[index, :]

        locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=rori[None, :], ray_directions=rdir[None, :])
        if verbose:
            print(locations, index_ray, index_tri)
            print(index_center, index_C)
        return idx, C, index_center, locations[0]

    def get_keypoints(self, n_eig=30, n_points=5, sym=False):
        evals, evecs = self.lapEigen(n_eig+1)
        pts_evecs = evecs[:, 1:] * np.exp(-2*0.1 * evals[1:]) # approximate geod distance
        tree = cKDTree(pts_evecs)
        _, center_index = tree.query(np.zeros(pts_evecs.shape[-1]), k=1)
        indices = fps_gpt_geod(self.vertices, self.faces, 10, center_index)[1:]
        areas = np.linalg.norm(self.surfel, axis=-1, keepdims=True)
        area = np.sqrt(areas.sum()/2)
        print(area)
        ## Looking for center + provide distances to the center
        
        print(center_index)
        solver = pp3d.MeshHeatMethodDistanceSolver(self.vertices, self.faces)
        norm_center = solver.compute_distance(center_index)
    
        ## Dist matrix just between selected points
        dist_mat_indices = np.zeros((len(indices), len(indices)))
        all_ii = np.arange(len(indices)) # just to easy code
        for ii, index in enumerate(indices):
            dist_mat_indices[ii, ii!=all_ii] = solver.compute_distance(index)[indices[indices != index]]
    
        ## Select points which : farthest from the center, delete their neighbors with dist < 0.5*area
        keep = []
        max_norm = norm_center.max()
        for ii, index in enumerate(indices[:n_points]):
            # distances_compar = (dist_mat_indices[ii, ii!=all_ii] + norm_center[ii]) / norm_center[indices[ii!=all_ii]]
            # min_dist = np.amin(np.abs(distances_compar - 1))
            # print(min_dist/norm_center[ii], min_dist)
            # if min_dist > 0.3:
            #     #print(ii, min_dist/norm_center[ii], norm_center[ii], dist_mat_indices[ii, ii!=all_ii], norm_center[indices[indices != index]], indices[7])
            keep.append(index)
        ## Add the index of the center
        ## Add the index of the center 
        if sym:
            mesh = tm.base.Trimesh(self.vertices, self.faces, process=False)  # Load your mesh
            # Given data
            index = 123  # Your starting point index
            direction = -mesh.vertex_normals[center_index]
            
            # Get the starting vertex position
            start_point = mesh.vertices[center_index]
            
            # Perform ray intersection
            locations, index_ray, index_tri = mesh.ray.intersects_location(
                ray_origins=start_point[None, :],  # Starting point
                ray_directions=direction[None, :]  # Direction of travel
            )
            
            # If intersections exist
            if len(locations) > 0:
                # Sort intersections by distance from start_point
                distances = np.linalg.norm(locations - start_point, axis=1)
                sorted_indices = np.argsort(distances)
            
                # Get the first valid intersection point beyond the start point
                #next_intersection = locations[sorted_indices]
                #print(f"Next intersection at: {next_intersection}")
                intersected_triangle = index_tri[sorted_indices[1]]
                print(f"Intersected Triangle Index: {mesh.faces[intersected_triangle]}")
                center_index = mesh.faces[intersected_triangle][0]
            else:
                print("No intersection found in the given direction.")
        keep.append(center_index)
        return keep


    # Computes surface volume
    def surfVolume(self):
        f = self.faces
        v = self.vertices
        t = v[f, :]
        vols = np.linalg.det(t) / 6
        return vols.sum(), vols

    def surfCenter(self):
        f = self.faces
        v = self.vertices
        center_infs = (v[f, :].sum(axis=1) / 4) * self.vols[:, np.newaxis]
        center = center_infs.sum(axis=0)
        return center

    def surfMoments(self):
        f = self.faces
        v = self.vertices
        vec_0 = v[f[:, 0], :] + v[f[:, 1], :]
        s_0 = vec_0[:, :, np.newaxis] * vec_0[:, np.newaxis, :]
        vec_1 = v[f[:, 0], :] + v[f[:, 2], :]
        s_1 = vec_1[:, :, np.newaxis] * vec_1[:, np.newaxis, :]
        vec_2 = v[f[:, 1], :] + v[f[:, 2], :]
        s_2 = vec_2[:, :, np.newaxis] * vec_2[:, np.newaxis, :]
        moments_inf = self.vols[:, np.newaxis, np.newaxis] * (1. / 20) * (s_0 + s_1 + s_2)
        return moments_inf.sum(axis=0)

    def surfF(self):
        f = self.faces
        v = self.vertices
        cent = (v[f, :].sum(axis=1)) / 4.
        F = self.vols[:, np.newaxis] * np.sign(cent) * (cent ** 2)
        return F.sum(axis=0)

    def surfU(self):
        vol = self.volume
        vertices = self.vertices / pow(vol, 1. / 3)
        vertices -= self.center
        self.updateVertices(vertices)
        moments = self.surfMoments()
        u, s, vh = np.linalg.svd(moments)
        F = self.surfF()
        return np.diag(np.sign(F)) @ u, s

    def surfEllipsoid(self, u, s, moments):
        coeff = pow(4 * np.pi / 15, 1. / 5) * pow(np.linalg.det(moments), -1. / 10)
        A = coeff * ((u * np.sqrt(s)) @ u.T)
        return u, A

    # Reads from .off file
    def readOFF(self, offfile):
        with open(offfile, 'r') as f:
            all_lines = f.readlines()
    
    
        n_vertices = int(all_lines[1].split()[0])
        n_faces = int(all_lines[1].split()[1])

        vertices_list = []
        for i in range(n_vertices):
            vertices_list.append([float(x) for x in all_lines[2+i].split()[:3]])

        faces_list = []
        for i in range(n_faces):
            # Be careful to convert to int. Otherwise, you can use np.array(faces_list).astype(np.int32)
            faces_list.append([int(x) for x in all_lines[2+i+n_vertices].split()[1:4]])
        self.faces = np.array(faces_list)
        self.vertices = np.array(vertices_list)
        self.computeCentersAreas()    # Reads from .byu file
    def readbyu(self, byufile):
        with open(byufile, 'r') as fbyu:
            ln0 = fbyu.readline()
            ln = ln0.split()
            # read header
            ncomponents = int(ln[0])  # number of components
            npoints = int(ln[1])  # number of vertices
            nfaces = int(ln[2])  # number of faces
            # fscanf(fbyu,'%d',1);		% number of edges
            # %ntest = fscanf(fbyu,'%d',1);		% number of edges
            for k in range(ncomponents):
                fbyu.readline()  # components (ignored)
            # read data
            self.vertices = np.empty([npoints, 3])
            k = -1
            while k < npoints - 1:
                ln = fbyu.readline().split()
                k = k + 1;
                self.vertices[k, 0] = float(ln[0])
                self.vertices[k, 1] = float(ln[1])
                self.vertices[k, 2] = float(ln[2])
                if len(ln) > 3:
                    k = k + 1;
                    self.vertices[k, 0] = float(ln[3])
                    self.vertices[k, 1] = float(ln[4])
                    self.vertices[k, 2] = float(ln[5])

            self.faces = np.empty([nfaces, 3])
            ln = fbyu.readline().split()
            kf = 0
            j = 0
            while ln:
                if kf >= nfaces:
                    break
                    # print(nfaces, kf, ln)
                for s in ln:
                    self.faces[kf, j] = int(sp.fabs(int(s)))
                    j = j + 1
                    if j == 3:
                        kf = kf + 1
                        j = 0
                ln = fbyu.readline().split()
        self.faces = np.int_(self.faces) - 1
        xDef1 = self.vertices[self.faces[:, 0], :]
        xDef2 = self.vertices[self.faces[:, 1], :]
        xDef3 = self.vertices[self.faces[:, 2], :]
        self.centers = (xDef1 + xDef2 + xDef3) / 3
        self.surfel = np.cross(xDef2 - xDef1, xDef3 - xDef1)

    # Saves in .byu format
    def savebyu(self, byufile):
        # FV = readbyu(byufile)
        # reads from a .byu file into matlab's face vertex structure FV

        with open(byufile, 'w') as fbyu:
            # copy header
            ncomponents = 1  # number of components
            npoints = self.vertices.shape[0]  # number of vertices
            nfaces = self.faces.shape[0]  # number of faces
            nedges = 3 * nfaces  # number of edges

            str = '{0: d} {1: d} {2: d} {3: d} 0\n'.format(ncomponents, npoints, nfaces, nedges)
            fbyu.write(str)
            str = '1 {0: d}\n'.format(nfaces)
            fbyu.write(str)

            k = -1
            while k < (npoints - 1):
                k = k + 1
                str = '{0: f} {1: f} {2: f} '.format(self.vertices[k, 0], self.vertices[k, 1], self.vertices[k, 2])
                fbyu.write(str)
                if k < (npoints - 1):
                    k = k + 1
                    str = '{0: f} {1: f} {2: f}\n'.format(self.vertices[k, 0], self.vertices[k, 1], self.vertices[k, 2])
                    fbyu.write(str)
                else:
                    fbyu.write('\n')

            j = 0
            for k in range(nfaces):
                for kk in (0, 1):
                    fbyu.write('{0: d} '.format(self.faces[k, kk] + 1))
                    j = j + 1
                    if j == 16:
                        fbyu.write('\n')
                        j = 0

                fbyu.write('{0: d} '.format(-self.faces[k, 2] - 1))
                j = j + 1
                if j == 16:
                    fbyu.write('\n')
                    j = 0

    def saveVTK(self, fileName, scalars=None, normals=None, tensors=None, scal_name='scalars', vectors=None,
                vect_name='vectors'):
        vf = vtkFields()
        # print(scalars)
        if not (scalars == None):
            vf.scalars.append(scal_name)
            vf.scalars.append(scalars)
        if not (vectors == None):
            vf.vectors.append(vect_name)
            vf.vectors.append(vectors)
        if not (normals == None):
            vf.normals.append('normals')
            vf.normals.append(normals)
        if not (tensors == None):
            vf.tensors.append('tensors')
            vf.tensors.append(tensors)
        self.saveVTK2(fileName, vf)

    # Saves in .vtk format
    def saveVTK2(self, fileName, vtkFields=None):
        F = self.faces;
        V = self.vertices;

        with open(fileName, 'w') as fvtkout:
            fvtkout.write('# vtk DataFile Version 3.0\nSurface Data\nASCII\nDATASET POLYDATA\n')
            fvtkout.write('\nPOINTS {0: d} float'.format(V.shape[0]))
            for ll in range(V.shape[0]):
                fvtkout.write('\n{0: f} {1: f} {2: f}'.format(V[ll, 0], V[ll, 1], V[ll, 2]))
            fvtkout.write('\nPOLYGONS {0:d} {1:d}'.format(F.shape[0], 4 * F.shape[0]))
            for ll in range(F.shape[0]):
                fvtkout.write('\n3 {0: d} {1: d} {2: d}'.format(F[ll, 0], F[ll, 1], F[ll, 2]))
            if not (vtkFields == None):
                wrote_pd_hdr = False
                if len(vtkFields.scalars) > 0:
                    if not wrote_pd_hdr:
                        fvtkout.write(('\nPOINT_DATA {0: d}').format(V.shape[0]))
                        wrote_pd_hdr = True
                    nf = len(vtkFields.scalars) / 2
                    for k in range(nf):
                        fvtkout.write('\nSCALARS ' + vtkFields.scalars[2 * k] + ' float 1\nLOOKUP_TABLE default')
                        for ll in range(V.shape[0]):
                            # print(scalars[ll])
                            fvtkout.write('\n {0: .5f}'.format(vtkFields.scalars[2 * k + 1][ll]))
                if len(vtkFields.vectors) > 0:
                    if not wrote_pd_hdr:
                        fvtkout.write(('\nPOINT_DATA {0: d}').format(V.shape[0]))
                        wrote_pd_hdr = True
                    nf = len(vtkFields.vectors) / 2
                    for k in range(nf):
                        fvtkout.write('\nVECTORS ' + vtkFields.vectors[2 * k] + ' float')
                        vectors = vtkFields.vectors[2 * k + 1]
                        for ll in range(V.shape[0]):
                            fvtkout.write(
                                '\n {0: .5f} {1: .5f} {2: .5f}'.format(vectors[ll, 0], vectors[ll, 1], vectors[ll, 2]))
                if len(vtkFields.normals) > 0:
                    if not wrote_pd_hdr:
                        fvtkout.write(('\nPOINT_DATA {0: d}').format(V.shape[0]))
                        wrote_pd_hdr = True
                    nf = len(vtkFields.normals) / 2
                    for k in range(nf):
                        fvtkout.write('\nNORMALS ' + vtkFields.normals[2 * k] + ' float')
                        vectors = vtkFields.normals[2 * k + 1]
                        for ll in range(V.shape[0]):
                            fvtkout.write(
                                '\n {0: .5f} {1: .5f} {2: .5f}'.format(vectors[ll, 0], vectors[ll, 1], vectors[ll, 2]))
                if len(vtkFields.tensors) > 0:
                    if not wrote_pd_hdr:
                        fvtkout.write(('\nPOINT_DATA {0: d}').format(V.shape[0]))
                        wrote_pd_hdr = True
                    nf = len(vtkFields.tensors) / 2
                    for k in range(nf):
                        fvtkout.write('\nTENSORS ' + vtkFields.tensors[2 * k] + ' float')
                        tensors = vtkFields.tensors[2 * k + 1]
                        for ll in range(V.shape[0]):
                            for kk in range(2):
                                fvtkout.write(
                                    '\n {0: .5f} {1: .5f} {2: .5f}'.format(tensors[ll, kk, 0], tensors[ll, kk, 1],
                                                                           tensors[ll, kk, 2]))
                fvtkout.write('\n')

    # Reads .vtk file
    def readVTK(self, fileName):
        if gotVTK:
            u = vtkPolyDataReader()
            u.SetFileName(fileName)
            u.Update()
            v = u.GetOutput()
            # print(v)
            npoints = int(v.GetNumberOfPoints())
            nfaces = int(v.GetNumberOfPolys())
            V = np.zeros([npoints, 3])
            for kk in range(npoints):
                V[kk, :] = np.array(v.GetPoint(kk))

            F = np.zeros([nfaces, 3])
            for kk in range(nfaces):
                c = v.GetCell(kk)
                for ll in range(3):
                    F[kk, ll] = c.GetPointId(ll)

            self.vertices = V
            self.faces = np.int_(F)
            xDef1 = self.vertices[self.faces[:, 0], :]
            xDef2 = self.vertices[self.faces[:, 1], :]
            xDef3 = self.vertices[self.faces[:, 2], :]
            self.centers = (xDef1 + xDef2 + xDef3) / 3
            self.surfel = np.cross(xDef2 - xDef1, xDef3 - xDef1)
        else:
            raise Exception('Cannot run readVTK without VTK')

    # Reads .obj file
    def readOBJ(self, fileName):
        if gotVTK:
            u = vtkOBJReader()
            u.SetFileName(fileName)
            u.Update()
            v = u.GetOutput()
            # print(v)
            npoints = int(v.GetNumberOfPoints())
            nfaces = int(v.GetNumberOfPolys())
            V = np.zeros([npoints, 3])
            for kk in range(npoints):
                V[kk, :] = np.array(v.GetPoint(kk))

            F = np.zeros([nfaces, 3])
            for kk in range(nfaces):
                c = v.GetCell(kk)
                for ll in range(3):
                    F[kk, ll] = c.GetPointId(ll)

            self.vertices = V
            self.faces = np.int_(F)
            xDef1 = self.vertices[self.faces[:, 0], :]
            xDef2 = self.vertices[self.faces[:, 1], :]
            xDef3 = self.vertices[self.faces[:, 2], :]
            self.centers = (xDef1 + xDef2 + xDef3) / 3
            self.surfel = np.cross(xDef2 - xDef1, xDef3 - xDef1)
        else:
            raise Exception('Cannot run readOBJ without VTK')

    def readPLY(self, fileName):
        if gotVTK:
            u = vtkPLYReader()
            u.SetFileName(fileName)
            u.Update()
            v = u.GetOutput()
            # print(v)
            npoints = int(v.GetNumberOfPoints())
            nfaces = int(v.GetNumberOfPolys())
            V = np.zeros([npoints, 3])
            for kk in range(npoints):
                V[kk, :] = np.array(v.GetPoint(kk))

            F = np.zeros([nfaces, 3])
            for kk in range(nfaces):
                c = v.GetCell(kk)
                for ll in range(3):
                    F[kk, ll] = c.GetPointId(ll)

            self.vertices = V
            self.faces = np.int_(F)
            xDef1 = self.vertices[self.faces[:, 0], :]
            xDef2 = self.vertices[self.faces[:, 1], :]
            xDef3 = self.vertices[self.faces[:, 2], :]
            self.centers = (xDef1 + xDef2 + xDef3) / 3
            self.surfel = np.cross(xDef2 - xDef1, xDef3 - xDef1)
        else:
            raise Exception('Cannot run readPLY without VTK')

    def readTRI(self, fileName):
        vertices = []
        faces = []
        with open(fileName, "r") as f:
            pithc = f.readlines()
            line_1 = pithc[0]
            n_pts, n_tri = line_1.split(' ')
            for i, line in enumerate(pithc[1:]):
                pouetage = line.split(' ')
                if i <= int(n_pts):
                    vertices.append([float(poupout) for poupout in pouetage[:3]])
                    # vertices.append([float(poupout) for poupout in pouetage[4:7]])
                else:
                    faces.append([int(poupout.split("\n")[0]) for poupout in pouetage[1:]])
        self.vertices = np.array(vertices)
        self.faces = np.array(faces)
        xDef1 = self.vertices[self.faces[:, 0], :]
        xDef2 = self.vertices[self.faces[:, 1], :]
        xDef3 = self.vertices[self.faces[:, 2], :]
        self.centers = (xDef1 + xDef2 + xDef3) / 3
        self.surfel = np.cross(xDef2 - xDef1, xDef3 - xDef1)

    def concatenate(self, fvl):
        nv = 0
        nf = 0
        for fv in fvl:
            nv += fv.vertices.shape[0]
            nf += fv.faces.shape[0]
        self.vertices = np.zeros([nv, 3])
        self.faces = np.zeros([nf, 3], dtype='int')

        nv0 = 0
        nf0 = 0
        for fv in fvl:
            nv = nv0 + fv.vertices.shape[0]
            nf = nf0 + fv.faces.shape[0]
            self.vertices[nv0:nv, :] = fv.vertices
            self.faces[nf0:nf, :] = fv.faces + nv0
            nv0 = nv
            nf0 = nf
        self.computeCentersAreas()

def get_surf_area(surf):
    areas = np.linalg.norm(surf.surfel, axis=-1)
    return areas.sum()/2

def centroid(surf):
    areas = np.linalg.norm(surf.surfel, axis=-1, keepdims=True)
    center = (surf.centers * areas).sum(axis=0) / areas.sum()
    return center, np.sqrt(areas.sum()/2)


def do_bbox_vertices(vertices):
    new_verts = np.zeros(vertices.shape)
    new_verts[:, 0] = (vertices[:, 0] - np.amin(vertices[:, 0]))/(np.amax(vertices[:, 0]) - np.amin(vertices[:, 0]))
    new_verts[:, 1] = (vertices[:, 1] - np.amin(vertices[:, 1]))/(np.amax(vertices[:, 1]) - np.amin(vertices[:, 1]))
    new_verts[:, 2] = (vertices[:, 2] - np.amin(vertices[:, 2]))/(np.amax(vertices[:, 1]) - np.amin(vertices[:, 2]))*0.5
    return new_verts

def opt_rot_surf(surf_1, surf_2, areas_c):
    center_1, _ = centroid(surf_1)
    pts_c1 = surf_1.centers - center_1
    center_2, _ = centroid(surf_2)
    pts_c2 = surf_2.centers - center_2 
    to_sum = pts_c1[:, :, np.newaxis] * pts_c2[:, np.newaxis, :]
    A = (to_sum * areas_c[:, :, np.newaxis]).sum(axis=0)
    #A = to_sum.sum(axis=0)
    u, _, v = np.linalg.svd(A)
    a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.sign(np.linalg.det(A))]])
    O = u @ a @ v
    return O.T

def srnf(surf):
    areas = np.linalg.norm(surf.surfel, axis=-1, keepdims=True) + 1e-8
    return surf.surfel/np.sqrt(areas)

def fps_gpt(points, num_samples):
    """
    Selects a subset of points using the Farthest Point Sampling algorithm.

    :param points: numpy array of shape (N, D), where N is the number of points and D is the dimension of each point
    :param num_samples: number of points to select
    :return: numpy array of indices of the selected points
    """
    if num_samples > len(points):
        raise ValueError("num_samples must be less than or equal to the number of points")

    # Initialize an array to hold the indices of the selected points
    selected_indices = np.zeros(num_samples, dtype=int)

    center = np.mean(points, axis=0)

    # Array to store the shortest distance of each point to a selected poin
    shortest_distances = np.full(len(points), np.inf)
    last_selected_point = center
    for i in range(1, num_samples):
        

        # Update the shortest distance of each point to the selected points
        for j, point in enumerate(points):
            distance = np.linalg.norm(point - last_selected_point)
            shortest_distances[j] = min(shortest_distances[j], distance)

        # Select the point that is farthest from all selected points
        selected_indices[i] = np.argmax(shortest_distances)
        last_selected_point = points[selected_indices[i]]
    return selected_indices


def fps_gpt_geod(points, faces, num_samples, index_start):
    """
    Selects a subset of points using the Farthest Point Sampling algorithm.

    :param points: numpy array of shape (N, D), where N is the number of points and D is the dimension of each point
    :param num_samples: number of points to select
    :return: numpy array of indices of the selected points
    """
    if num_samples > len(points):
        raise ValueError("num_samples must be less than or equal to the number of points")

    # Initialize an array to hold the indices of the selected points
    selected_indices = np.zeros(num_samples, dtype=int)

    # Array to store the shortest distance of each point to a selected poin
    shortest_distances = np.full(len(points), np.inf)
    selected_indices[0] = index_start
    distances = np.zeros((num_samples, len(points)))
    solver = pp3d.MeshHeatMethodDistanceSolver(points, faces)
    for i in range(1, num_samples):
        distances[i] = (solver.compute_distance(selected_indices[i-1]))

        # Update the shortest distance of each point to the selected points
        for j, point in enumerate(points):
            distance = distances[i][j]
            shortest_distances[j] = min(shortest_distances[j], distance)

        # Select the point that is farthest from all selected points
        selected_indices[i] = np.argmax(shortest_distances)
    return selected_indices

# class MultiSurface:
#     def __init__(self, pattern):
#         self.surf = []
#         files = glob.glob(pattern)
#         for f in files:
#             self.surf.append(Surface(filename=f))
