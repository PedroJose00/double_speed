from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem

# adiciono velocidad a la BMU
# +
import math
import ufl
import dolfinx 
import csv
import numpy as np
from dolfinx import la
from dolfinx.fem import (Expression, Function, FunctionSpace, dirichletbc,
                         form, functionspace, locate_dofs_topological, locate_dofs_geometrical)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               set_bc)
from dolfinx.io import XDMFFile
from dolfinx.mesh import (CellType, GhostMode, create_box,
                          locate_entities_boundary)
from ufl import dx, grad, inner, tr, det , ln

import time
import pandas as pd

# Marcar el tiempo de inicio
start_time = time.time()
#time.sleep(1)  # Simula el cálculo pausando la ejecución por 1 segundo


from scipy.interpolate import griddata, LinearNDInterpolator

from dolfinx.io import XDMFFile
from dolfinx.mesh import meshtags_from_entities
from dolfinx.cpp.mesh import cell_entity_type
from dolfinx.io import distribute_entity_data
from dolfinx.graph import adjacencylist
from dolfinx.mesh import create_mesh
from dolfinx.cpp.mesh import to_type
from dolfinx.cpp.io import perm_gmsh
import numpy
import meshio
from dolfinx.io import gmshio
from mpi4py import MPI
from dolfinx.io.gmshio import model_to_mesh
import numpy as np
import gmsh
import warnings
import meshio
warnings.filterwarnings("ignore")
dtype = PETSc.ScalarType  # type: ignore



dtype = PETSc.ScalarType  # type: ignore
# Suprimir advertencias (por ejemplo, de Gmsh cuando se trabaja con archivos STEP)
warnings.filterwarnings("ignore")





def build_nullspace(V: FunctionSpace):
    """Build PETSc nullspace for 3D elasticity"""

    # Create vectors that will span the nullspace
    bs = V.dofmap.index_map_bs
    length0 = V.dofmap.index_map.size_local
    basis = [la.vector(V.dofmap.index_map, bs=bs, dtype=dtype) for i in range(6)]
    b = [b.array for b in basis]

    # Get dof indices for each subspace (x, y and z dofs)
    dofs = [V.sub(i).dofmap.list.flatten() for i in range(3)]

    # Set the three translational rigid body modes
    for i in range(3):
        b[i][dofs[i]] = 1.0

    # Set the three rotational rigid body modes
    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.flatten()
    x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
    b[3][dofs[0]] = -x1
    b[3][dofs[1]] = x0
    b[4][dofs[0]] = x2
    b[4][dofs[2]] = -x0
    b[5][dofs[2]] = x1
    b[5][dofs[1]] = -x2

    _basis = [x._cpp_object for x in basis]
    dolfinx.cpp.la.orthonormalize(_basis)
    assert dolfinx.cpp.la.is_orthonormal(_basis)

    basis_petsc = [PETSc.Vec().createWithArray(x[:bs * length0], bsize=3, comm=V.mesh.comm) for x in b]  # type: ignore
    # Suponiendo que tienes un espacio de funciones llamado 'V'
    dof_per_node = V.dofmap.index_map_bs

# Imprimir la cantidad de DOF por nodo
#    print("Cantidad de DOF por nodo:", dof_per_node)
    return PETSc.NullSpace().create(vectors=basis_petsc)  # type: ignore

#----

atol=1e-3



def calculate_triangle_area(node_coords):
    p1, p2, p3 = node_coords
    return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))



# Inicializar Gmsh
gmsh.initialize()

# Crear un nuevo modelo
gmsh.model.add("modelo_3d")

# Cargar archivo STEP
archivo_step = "001.step"  # Reemplaza con la ruta a tu archivo STEP
gmsh.merge(archivo_step)

# Sincronizar para asegurar que la geometría se cargue en el modelo
gmsh.model.occ.synchronize()

volumes = gmsh.model.getEntities(dim=3)
bone_marker = 11
gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], bone_marker)
gmsh.model.setPhysicalName(volumes[0][0], bone_marker, "bone_marker")
#______________________________

surfaces = gmsh.model.getEntities(dim=2)

# Asignar un identificador único a cada superficie
for i, surface in enumerate(surfaces, start=1):
    # Crear un grupo físico para la superficie actual
    group_id = gmsh.model.addPhysicalGroup(2, [surface[1]])
    # Opcional: asignar un nombre al grupo físico
    gmsh.model.setPhysicalName(2, group_id, f"Surface_{i}")


# Definir el tamaño de los elementos de la malla
# Esto puede variar dependiendo de la geometría y los requerimientos de la malla
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 1.5)

# Generar la malla de volumen (tetraedros)
gmsh.model.mesh.generate(3)

# Guardar la malla en un archivo
#gmsh.write("bone_mesh_1.5.msh")   ---->>>>>>>
 

# Obtener los tipos de elementos y sus tags
element_types, element_tags, node_tags = gmsh.model.mesh.getElements()

# # Iterar sobre los tipos de elementos y sus tags
# for elem_type, tags in zip(element_types, element_tags):
#     for tag in tags:
#         print(f"Elemento del tipo {elem_type}, ID: {tag}")

# gmsh.write("mesh3D.msh")
x = gmshio.extract_geometry(gmsh.model)
topologies = gmshio.extract_topology_and_markers(gmsh.model)

# Asumiendo que estás trabajando con una malla 2D con elementos triangulares
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
element_types, element_tags, node_tags_elem = gmsh.model.mesh.getElements()


# mesh = meshio.read("malla_salida10000.msh")

#     # Información de la malla
# num_nodes = len(mesh.points)
# num_elements = sum(len(cell.data) for cell in mesh.cells)
# cell_types = set(cell.type for cell in mesh.cells)
# print(f"Número de Nodos: {num_nodes}")
# print(f"Número de Elementos: {num_elements}")
# print(f"Tipos de Elementos: {cell_types}")
#___________________________________________________________________________


# Get information about each cell type from the msh files
num_cell_types = len(topologies.keys())
cell_information = {}
cell_dimensions = numpy.zeros(num_cell_types, dtype=numpy.int32)
for i, element in enumerate(topologies.keys()):
    properties = gmsh.model.mesh.getElementProperties(element)
    name, dim, order, num_nodes, local_coords, _ = properties
    cell_information[i] = {"id": element, "dim": dim,
                           "num_nodes": num_nodes}
    cell_dimensions[i] = dim


# Sort elements by ascending dimension
perm_sort = numpy.argsort(cell_dimensions)

#___________________________________________________________________________

#___________________________________________________________________________

cell_id = cell_information[perm_sort[-1]]["id"]
cells = numpy.asarray(topologies[cell_id]["topology"], dtype=numpy.int64)
ufl_domain = gmshio.ufl_mesh(cell_id, 3)

# #___________________________________________________________________________

#___________________________________________________________________________


num_nodes = cell_information[perm_sort[-1]]["num_nodes"]
gmsh_cell_perm = perm_gmsh(to_type(str(ufl_domain.ufl_cell())), num_nodes)
cells = cells[:, gmsh_cell_perm]

#___________________________________________________________________________

mesh = create_mesh(MPI.COMM_SELF, cells, x, ufl_domain)
# Obtener las coordenadas de todos los vértices de la malla
coordinates = mesh.geometry.x
#filename = "mesh_coordinates.txt"

# # Escribir las coordenadas en el archivo
# with open(filename, 'w') as file:
#     for i, coord in enumerate(coordinates):
#         file.write(f"Vértice {i}: {coord}\n")

# print(f"Las coordenadas de la malla se han guardado en '{filename}'")
#___________________________________________________________________________

# Create MeshTags for cell data
cell_values = numpy.asarray(
    topologies[cell_id]["cell_data"], dtype=numpy.int32)
local_entities, local_values = distribute_entity_data(
    mesh, mesh.topology.dim, cells, cell_values)
mesh.topology.create_connectivity(mesh.topology.dim, 0)
adj = adjacencylist(local_entities)
ct = meshtags_from_entities(mesh, mesh.topology.dim, adj, local_values)
ct.name = "Cell tags"

# Create MeshTags for facets
# Permute facets from MSH to DOLFINx ordering
# FIXME: This does not work for prism meshes
facet_type = cell_entity_type(
    to_type(str(ufl_domain.ufl_cell())), mesh.topology.dim - 1, 0)
gmsh_facet_id = cell_information[perm_sort[-2]]["id"]
num_facet_nodes = cell_information[perm_sort[-2]]["num_nodes"]
gmsh_facet_perm = perm_gmsh(facet_type, num_facet_nodes)
marked_facets = numpy.asarray(
    topologies[gmsh_facet_id]["topology"], dtype=numpy.int64)
facet_values = numpy.asarray(
    topologies[gmsh_facet_id]["cell_data"], dtype=numpy.int32)
marked_facets = marked_facets[:, gmsh_facet_perm]
local_entities, local_values = distribute_entity_data(
    mesh, mesh.topology.dim - 1, marked_facets, facet_values)
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
adj = adjacencylist(local_entities)
ft = meshtags_from_entities(mesh, mesh.topology.dim - 1, adj, local_values)
ft.name = "Facet tags"

# Output DOLFINx meshes to file

# with XDMFFile(MPI.COMM_WORLD, "bone_out_1.5.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)
#     xdmf.write_meshtags(ft, mesh.geometry)
#     xdmf.write_meshtags(ct, mesh.geometry)
gmsh.finalize()


# Asumimos que el archivo se llama 'propiedades.csv' y ya está en la ubicación correcta
archivo_propiedades = './propiedades_002.csv'

# Leer el archivo
df = pd.read_csv(archivo_propiedades)

# Asumiendo que el archivo tiene columnas 'x', 'y', 'z', 'densidad', 'E', 'nu'
puntos = df[['X', 'Y', 'Z']].to_numpy()

#------------------------------------------------------------------------------
###  --->>>> valor de la densidad aparente esta kg/mm^3
#------------------------------------------------------------------------------
df['DENS'] = df['DENS']*1e-9
df_densidades = df['DENS'].to_numpy()
#------------------------------------------------------------------------------
#BVTV= 2.05e-6/densidades  
#modulos_E = 84370*BVTV**2.58*0.70**2.74*1e12
### EL modulo de elasticidad esta N/mm^2
#------------------------------------------------------------------------------
df['E']=df['E']*1e3            
df_modulos_E = df['E'].to_numpy()
#------------------------------------------------------------------------------
# El valor del módulo de poisson son es adimensional
df_coeficientes_nu = df['NU'].to_numpy()
#------------------------------------------------------------------------------
# los valores por defecto inicualmente la mediana del valor
#------------------------------------------------------------------------------
valor_por_defecto = np.median(df_densidades)  # O np.mean(valores_densidad)
valor_por_defecto_e = np.median(df_modulos_E)   
valor_por_defecto_nu = np.median(df_coeficientes_nu)   
#------------------------------------------------------------------------------

# print(df[['E', 'DENS']].head())
# df.to_csv('datos_ajustados.csv', index=False)

print(df[['E', 'DENS']].head())
#df.to_csv('./01_03_24/modulos.csv', index=False)

alpha_ini = 0.6 # Fracción inicial de cenizas

W =fem.FunctionSpace(mesh,("DG",0)) 

fem_function_densidad_0= fem.Function(W, name="Densidad")
fem_function_modulo_E_0 = fem.Function(W,name="Modulo_young")
fem_function_coeficientes_nu_0 = fem.Function(W,name="Poisson")
fem_function_alpha = fem.Function(W, name="Alpha")
fem_function_BVTV = fem.Function(W,name="BVTV")
fem_function_array_error= fem.Function(W, name="Error") 
fem_function_densidad_material=fem.Function(W,name="Densidad_material")


# Crear un espacio de funciones para el tensor como un espacio vectorial
S = fem.VectorFunctionSpace(mesh, ("DG", 0), dim=4)

# Crear una función para almacenar los valores del tensor de esfuerzos
sigma_field = fem.Function(S)



#sigma_bar_i= fem.Function(W,name="sigma_bar_i")
#psi = fem.Function(W,name="psi")
# Ahora, para imprimir estos valores en un archivo, puedes usar el método to_csv de Pandas


#df[['modulos_E']].to_csv('./out_properties/modulos_E_ajustados.csv', index=False)

# Obtener las coordenadas de los centros de los elementos o nodos de la malla Fenicsx
mesh_coordinates = W.tabulate_dof_coordinates()  # Asumiendo que W es tu espacio de funciones DG0

# Extracción de valores de la función FEniCS/DOLFINx (Este es un paso conceptual, ajusta según sea necesario)
#valores_densidad = [fem_function_densidad_0.eval(punto) for punto in puntos]  # Ajusta esta línea según sea necesario


# Crear el interpolador
linear_interp_densidad_0 = LinearNDInterpolator( puntos , df_densidades)
linear_interp_modulo_E_0 = LinearNDInterpolator(puntos , df_modulos_E)
#interpolador_nu = LinearNDInterpolator(puntos, coeficientes_nu)


# Usar el interpolador para obtener valores de densidad en las coordenadas de la malla
mesh_value_densidad_0 = linear_interp_densidad_0(mesh_coordinates)
mesh_value_modulo_E_0 = linear_interp_modulo_E_0(mesh_coordinates)
#valores_interpolados_nu = interpolador_nu(mesh_coordinates)

# Manejar posibles NaNs después de la interpolación
mesh_value_densidad_0  = np.nan_to_num(mesh_value_densidad_0, nan=valor_por_defecto)
mesh_value_modulo_E_0  = np.nan_to_num(mesh_value_modulo_E_0 , nan=valor_por_defecto_e)
#valores_interpolados_nu = np.nan_to_num(valores_interpolados, nan=valor_por_defecto_nu)

# Asumiendo que densidad_h es tu función Fenicsx en el espacio DG0
fem_function_densidad_0.vector.setArray(mesh_value_densidad_0)
fem_function_modulo_E_0.vector.setArray(mesh_value_modulo_E_0)
array_modulo_E_0 = fem_function_modulo_E_0.x.array[:]
array_densidad_0= fem_function_densidad_0.x.array[:]





array_densidad = fem_function_densidad_0.x.array[:]

# el valor de la densidad para calcular el modulo debe estar en gr/cm3
array_densidad_gr_cm3 = array_densidad*1e6

# Calcula el valor promedio de array_densidad
promedio_densidad = np.mean(array_densidad_gr_cm3)

print("El valor promedio de la densidad es:", promedio_densidad )

array_BVTV = array_densidad_gr_cm3 /(1.41 +1.29*alpha_ini)
print("00__Hay NaN en expression_array_BVTV:", np.isnan(array_BVTV).any())

fem_function_BVTV.x.array[:]= array_BVTV
                        

with XDMFFile(mesh.comm, "./20_01_05_24/fem_function_BVTV.xdmf", "w") as file:
          file.write_mesh(mesh)
          file.write_function(fem_function_BVTV ,0.0)




array_error= np.empty_like(array_BVTV)
array_rdot = np.empty_like(array_error)
array_alpha_t = np.empty_like(array_BVTV)
array_densidad_material = np.empty_like(array_BVTV)
array_formacion_vb = np.zeros_like(array_BVTV)



array_porosidad = 1 - array_BVTV

# Calcula el valor promedio de array_densidad
promedio_porosidad = np.mean(array_porosidad)
print("El valor promedio de la porosidad es:", promedio_porosidad)

#---------------------------------------------------------------------------------------
# El cálculo de la porosidad proviene  del calculo del valor de la superficie específica
# --------------------------------------------------------------------------------------
array_porosidad = 1 - array_BVTV



# with XDMFFile(mesh.comm, "./09_04_24/array_BMU.xdmf", "w") as xdmf:
#                 # Guardar la función BMU en el archivo
#     xdmf.write_mesh(mesh)    
#     xdmf.write_function(fem_function_N_BMU , 0.0)  # El segundo 
# with XDMFFile(mesh.comm, "./09_04_24/array_s.xdmf", "w") as xdmf:
#                 # Guardar la funciónarray_s en el archivo
#     xdmf.write_mesh(mesh)    
#     xdmf.write_function(fem_function_array_s , 0.0)  # El segundo 


# Paso 2: Realizar la operación deseada sobre estos arrays para calcular E
                       


# __________---------------------------------------------------------------------------------------------------------
array_modulo_E_f = 84370*(array_BVTV ** 2.58) * (alpha_ini ** 2.74)*1e3
promedio_E_f = np.mean(array_modulo_E_f)
print("El valor promedio de la E_f es:", promedio_E_f)
promedio_E_0 = np.mean(array_modulo_E_0)
print("El valor promedio de la E_0 es:", promedio_E_0)
array_damage = 1- (array_modulo_E_f/ array_modulo_E_0)
promedio_daño = np.mean(array_damage)
print("El valor promedio de daño es:", promedio_daño )


print("Dimensiones de la matriz de porosidad:", array_porosidad.shape)
print("Dimensiones de la matriz de elasticidad:", array_modulo_E_f.shape)

# =================================================================================
# Debo calcular la matriz de porosidad.
# Los valores principales de la matriz de porosidad representan el tensor de tejido
# array_densidad_gr_cm3 = array_densidad*1e6
# _-------------------------------------------------------------------------------


fem_function_damage= fem.Function(W)
fem_function_modulo_E_f = fem.Function(W)
fem_function_densidad = fem.Function(W,name="densidad_grcm3")
fem_function_vr_dot = fem.Function(W,name="osteoclast_activity")
fem_function_vb_dot = fem.Function(W,name="osteoblast_activity")
fem_function_densidad_gr_cm3 = fem.Function(W,name= "densidad_grcm3")
fem_function_array_psi= fem.Function(W, name="array_psi")
fem_function_array_rdot= fem.Function(W,name="rdot")
fem_function_array_vm = fem.Function(W,name="array_vm")
#fem_function_densidad_material = fem.Function(W,"densidad_material")
sigma_vm_h = Function(W,name="Von_misses")
strain_ = Function(W,name="Strain")
value_sigma_h = Function(W,name="Stress")

fem_function_modulo_E_f.x.array[:] = array_modulo_E_f
fem_function_densidad_gr_cm3.x.array[:] = array_densidad_gr_cm3
fem_function_densidad.x.array[:] = array_densidad_gr_cm3 *1e-6

#=======================================================================================   
# ----------------------------------------------------------------------------------
# aca se pueden modulos y densidades iniciales
# ----------------------------------------------------------------------------------
with XDMFFile(mesh.comm, "./20_01_05_24/fem_function_modulo_E_0.xdmf", "w") as file:
          file.write_mesh(mesh)
          file.write_function(fem_function_modulo_E_f ,0.0)


with XDMFFile(mesh.comm, "./20_01_05_24/fem_function_densidad_0.xdmf", "w") as file:
          file.write_mesh(mesh)
          file.write_function(fem_function_densidad_gr_cm3 ,0.0)
# ----------------------------------------------------------------------------------
#=======================================================================================   


x = ufl.SpatialCoordinate(mesh)
f = ufl.as_vector((0.0, 0.0,-fem_function_densidad*9810))

# Define the elasticity parameters and create a function that computes
# an expression for the stress given a displacement field.

#=======================================================================================   
nu=0.3
# Al parecer tambien hay un error en el calculo de mu no es + sino -
μ = fem_function_modulo_E_f / (2.0 * (1.0 + nu))
λ = fem_function_modulo_E_f * nu/ ((1.0 + nu) * (1.0 - 2.0 * nu)) 
# revisar el valor de lamnda pues al parecer hay un error  
def σ(v):
    """Return an expression for the stress σ given a displacement field"""
    return 2.0 * μ * ufl.sym(grad(v)) + λ * ufl.tr(ufl.sym(grad(v))) * ufl.Identity(3)

def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

#=======================================================================================   

V = functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = form(inner(σ(u), grad(v)) * dx)
L = form(inner(f, v) * dx)



#----------------------------------------------------------------------------
# Asignacion de la condición frontera 08/02/24 - R-ninety
#----------------------------------------------------------------------------
def boundary_z(x):
    return np.logical_and(x[2] >= 36, x[2] <= 37)

# Ubicar los grados de libertad en las caras con z entre 36 y 37
dofs = dolfinx.fem.locate_dofs_geometrical(V, boundary_z)

# Definir la condición de frontera como desplazamiento 0 en todas las direcciones
u_D = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0, 0, 0)))

# Aplicar la condición de frontera
bc = dolfinx.fem.dirichletbc(u_D, dofs, V)
#----------------------------------------------------------------------------

# ## Assemble and solve
#
# The bilinear form `a` is assembled into a matrix `A`, with
# modifications for the Dirichlet boundary conditions. The call
# `A.assemble()` completes any parallel communication required to
# compute the matrix.

# +
A = assemble_matrix(a, bcs=[bc])
A.assemble()
# -----

# Las coordenadas de los siguientes puntos:
    # 138- 87.7431, 20.6717, 224.6261
    # 140- 85.6372, 33.9296, 226.9851
    # 141- 77.7509, 23.4239, 216.3084
    # 143- 78.43, 17.2494, 209.5734
    # 148- 73.5285, 30.5294, 203.1597






# Coordenadas de los puntos donde se aplicarán las cargas puntuales
point_138 = np.array([87.7431, 20.6717, 224.6261],dtype=np.float64)
point_140 = np.array([85.6372, 33.9296, 226.9851],dtype=np.float64)
point_141 = np.array([77.7509, 23.4239, 216.3084],dtype=np.float64)
point_143 = np.array([78.43, 17.2494, 209.5734],dtype=np.float64)
point_148 = np.array([73.5285, 30.5294, 203.1597],dtype=np.float64)
point_52 = np.array([124.2809, 54.4087,180.5598],dtype=np.float64)

# Cargas en la dirección Z y X

def marker_52(x):
    # Asegurarse de que x se compara correctamente con las coordenadas de point_138
    close_x = np.isclose(x[0], point_52[0], atol=atol)
    close_y = np.isclose(x[1], point_52[1], atol=atol)
    close_z = np.isclose(x[2], point_52[2], atol=atol)
    return np.logical_and(np.logical_and(close_x, close_y), close_z)

def marker_138(x):
    # Asegurarse de que x se compara correctamente con las coordenadas de point_138
    close_x = np.isclose(x[0], point_138[0], atol=atol)
    close_y = np.isclose(x[1], point_138[1], atol=atol)
    close_z = np.isclose(x[2], point_138[2], atol=atol)
    return np.logical_and(np.logical_and(close_x, close_y), close_z)

def marker_140(x):
    # Asegurarse de que x se compara correctamente con las coordenadas de point_138
    close_x = np.isclose(x[0], point_140[0], atol=atol)
    close_y = np.isclose(x[1], point_140[1], atol=atol)
    close_z = np.isclose(x[2], point_140[2], atol=atol)
    return np.logical_and(np.logical_and(close_x, close_y), close_z)
def marker_141(x):
    # Asegurarse de que x se compara correctamente con las coordenadas de point_138
    close_x = np.isclose(x[0], point_141[0], atol=atol)
    close_y = np.isclose(x[1], point_141[1], atol=atol)
    close_z = np.isclose(x[2], point_141[2], atol=atol)
    return np.logical_and(np.logical_and(close_x, close_y), close_z)
def marker_143(x):
    # Asegurarse de que x se compara correctamente con las coordenadas de point_138
    close_x = np.isclose(x[0], point_143[0], atol=atol)
    close_y = np.isclose(x[1], point_143[1], atol=atol)
    close_z = np.isclose(x[2], point_143[2], atol=atol)
    return np.logical_and(np.logical_and(close_x, close_y), close_z)

def marker_148(x):
    # Asegurarse de que x se compara correctamente con las coordenadas de point_138
    close_x = np.isclose(x[0], point_148[0], atol=atol)
    close_y = np.isclose(x[1], point_148[1], atol=atol)
    close_z = np.isclose(x[2], point_148[2], atol=atol)
    return np.logical_and(np.logical_and(close_x, close_y), close_z)



b = assemble_vector(L)

# Localiza los grados de libertad en el vértice
vertices_52 = locate_dofs_geometrical(V, marker_138)
dofs_52_x = (vertices_52[0])*3 + 0  # Grado de libertad en la dirección y
dofs_52_z = (vertices_52[0])*3 + 2  # Grado de libertad en la dirección z


vertices_138 = locate_dofs_geometrical(V, marker_138)
dofs_138_x = (vertices_138[0])*3 + 0  # Grado de libertad en la dirección y
dofs_138_z = (vertices_138[0])*3 + 2  # Grado de libertad en la dirección z

vertices_140 = locate_dofs_geometrical(V, marker_140)
dofs_140_x = (vertices_140[0])*3 + 0  # Grado de libertad en la dirección y
dofs_140_z = (vertices_140[0])*3 + 2  # Grado de libertad en la dirección z

vertices_141 = locate_dofs_geometrical(V, marker_141)
dofs_141_x = (vertices_141[0])*3 + 0  # Grado de libertad en la dirección y
dofs_141_z = (vertices_141[0])*3 + 2  # Grado de libertad en la dirección z

vertices_143 = locate_dofs_geometrical(V, marker_143)
dofs_143_x = (vertices_143[0])*3 + 0  # Grado de libertad en la dirección y
dofs_143_z = (vertices_143[0])*3 + 2  # Grado de libertad en la dirección z

vertices_148 = locate_dofs_geometrical(V, marker_148)
dofs_148_x = (vertices_148[0])*3 + 0  # Grado de libertad en la dirección y
dofs_148_z = (vertices_148[0])*3 + 2  # Grado de libertad en la dirección z



print("El valor de los DOFs en X y en Z en la  carga:", dofs_138_x, dofs_138_z)
# Magnitudes de las cargas puntuales
P_x = 0  # Carga en la dirección y 
P_z = -2317   # Carga en la dirección z
P_z2= 713       # Carga del abductor en z
    # Aplicar las cargas puntuales
with b.localForm() as loc_b:
    loc_b.setValues(dofs_138_x, P_x)
    loc_b.setValues(dofs_138_z, P_z)
    loc_b.setValues(dofs_140_x, P_x)
    loc_b.setValues(dofs_140_z, P_z) 
    loc_b.setValues(dofs_141_x, P_x)
    loc_b.setValues(dofs_141_z, P_z) 
    loc_b.setValues(dofs_143_x, P_x)
    loc_b.setValues(dofs_143_z, P_z) 
    loc_b.setValues(dofs_148_x, P_x)
    loc_b.setValues(dofs_148_z, P_z)
    loc_b.setValues(dofs_52_x, P_x)
    loc_b.setValues(dofs_52_z, P_z2)

apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
set_bc(b, [bc])


ns = build_nullspace(V)
A.setNearNullSpace(ns)
A.setOption(PETSc.Mat.Option.SPD, True)  # type: ignore

# Set PETSc solver options, create a PETSc Krylov solver, and attach the
# matrix `A` to the solver:

# +
# Set solver options
opts = PETSc.Options()  # type: ignore
opts["ksp_type"] = "cg"
opts["ksp_rtol"] = 1.0e-8
opts["pc_type"] = "gamg"

# Use Chebyshev smoothing for multigrid
opts["mg_levels_ksp_type"] = "chebyshev"
opts["mg_levels_pc_type"] = "jacobi"

# Improve estimate of eigenvalues for Chebyshev smoothing
opts["mg_levels_ksp_chebyshev_esteig_steps"] = 10

#----



# Create PETSc Krylov solver and turn convergence monitoring on
solver = PETSc.KSP().create(mesh.comm)  # type: ignore
solver.setFromOptions()

# Set matrix operator
solver.setOperators(A)
# -
#----
# Create a solution {py:class}`Function<dolfinx.fem.Function>` `uh` and
# solve:

# +
uh = Function(V, name="desplazamiento - (mm)")

# Set a monitor, solve linear system, and display the solver
# configuration
#solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}"))
solver.solve(b, uh.vector)


# Scatter forward the solution vector to update ghost values
uh.x.scatter_forward()
# -

# ## Post-processing
#
# The computed solution is now post-processed. Expressions for the
# deviatoric and Von Mises stress are defined:

# +


# Calcular la energía de deformación
U = 0.5 * ufl.inner(σ(uh),epsilon(uh)) * ufl.dx

# Ensamblar la energía de deformación ----<< 09/02/24 ---<<< OK
energy = fem.assemble_scalar(fem.form(U)) 

print("Energía en ciclo_o:",energy)                       


# ------------------------------------------------------------------
# Hay que calcular las propiedades modulo de elasticidad
alpha_ini = 0.6 # Fracción inicial de cenizas
alpha_0 = 0.45  # Fraccion minima de cenizas
alpha_max = 0.7 # Fracción máxima de cenizas
k_alpha = 6     # Secondary Period of mineralization


#------------------------------------------------------------------------------
# Variables del modelo de daño 
# --------------------------------------------------------------------------------------

damage_expression =0
rho_app = 1.75e-6
rho_app_value = fem.Constant(mesh,rho_app)
constant_c = 2.5e-3    # Parámetro de activacion de la curva de deformacion 
constant_a = 50         # exponente de activación de daño
escalar_fbio = 0.2     # Factor de frecuencia biológica 
Carter_1989_A =0.05194026
Carter_1989_B =-0.057852   
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
# Todo esto viene de (A-R-F) activation, resorption, formation
# para un total de 96 days TR + TI + TF
 #---------------------------------------------------------------------------------------
escalar_TR = 24     # 24 days  resorption time
escalar_TI = 8      # 8 days
escalar_TF = 64     # 64 days
escalar_fc= 1       # Constante balance focal para osteoclastos
escalar_fb= 1       # COnstante balance focal para osteoblastos
# -------------------------------------------------------------------------------------
# Registro de datos en el tiempo para mostrar resultados
# ------------------------------------------------------------------------------------
escalar_contador_total_dias = 0
escalar_contador_total_dias_modelo = 0
promedios_damage = np.zeros(1000)
promedios_densidad = np.zeros(1000)
promedios_modulo_elasticidad = np.zeros(1000)
promedios_BVTV = np.zeros(1000)
promedios_rho_cortical_gr_cm_3 = np.zeros(1000)
promedios_alpha= np.zeros(1000)
escalar_energy = np.zeros(1000)
promedios_rho_tejido= np.zeros(1000)

# ----------------------------------------------------------------------------------
# para calcular alpha es necesario el valor de las constantes
#  array_alpha = escalar_rho_mineral*array_vm/escalar_rho_mineral*array_vm + escalar_v_org* escalar_rho_org
# ----------------------------------------------------------------------------------
escalar_rho_mineral = 3.2    # este valor esta en gr/cm^3
escalar_rho_org = 1.1       # este valor esta en gr/cm^3
escalar_rho_water = 1       # este valor esta en gr/cm^3
escalar_v_org = 3/7

escalar_vm_prim = 0.121     # volumen especifico al final dela fase primaria
escalar_vm_max = 0.442       #volumen espcifico máximo de contenido de calcio
escalar_t_prim = 10         # longitud de la fase primaria
escalar_t_nm = 12           # lag time de mineralización
# ----------------------------------------------------------------------------------

escalar_area_cortical = 4.3e-3
escalar_area_trabecular = 1.9e-3

array_COPY_BVTV =np.empty_like(array_BVTV)

#---------------------------------------------------------------------------------------
#  Velocidad de la BMU
escalar_V_bmu = 40e-3 # 40 µm por día 
# Para la fase 1-2-.
escalar_tmp_ar1 = 7.876e-6  # Este valor debe de estar en days/mm[2] Valor para fase R1
escalar_tmp_ar2 = 0.004536  # Este valor debe de estar en days/mm[2] Valor para fase Formacion fase 3 
escalar_tmp_af4 = 0.01749 # Éste valor representa la constante days/mm[2] para fase F4
escalar_tmp_ar5 = 0.004536 # este valor para calcular πdo^2/4
escalar_a_hemi = 0.005862 # area del  hemiosteon
# --------------------------------------------------------------------------------------

# Variables del modelo de crecimiento de daño. Hay que tener en cuenta que hay 2 situaciones
# Una función que trabaja tensión y otra función que trabaja a compresión, acaso detallan
#  las variables de cada una de las dos situaciones tensión y compresion. 
# -------------------------------------------------------------------------------------
escalar_delta_1 =10.3
escalar_delta_2 = 14.1

# --------------------------------------------------------------------------------------





start_time = time.time()

#---------------------------------------------------------------------------------
# Crear un array de ceros del tamaño adecuado para el espacio de funciones
# El tamaño del array debe coincidir con el número de grados de libertad (DoFs) del espacio de funciones
# inicializacion del arreglo de la funcion daño
#---------------------------------------------------------------------------------

expression_array_dot_vr = np.zeros(W.dofmap.index_map.size_local)
expression_array_dot_vb = np.zeros(W.dofmap.index_map.size_local)


#---------------------------------------------------------------------------------
# Calculo de parametros iniciales de variacion de la BMU
# El calculo se realiza con los valores iniciales de todas las variables
#---------------------------------------------------------------------------------
array_densidad = fem_function_densidad_0.x.array

# Calcula el valor promedio de array_densidad


array_densidad_gr_cm_3= array_densidad*1e6
promedio_densidad_gr_cm3 = np.mean(array_densidad_gr_cm3)

print("El valor promedio de la densidad gr/cm3 es:", promedio_densidad_gr_cm3)
#print("00__Hay NaN en expression_array_BVTV:", np.isnan(array_BVTV).any())




condicion_cortical= array_BVTV >= 0.31
condicion_trabecular = array_BVTV < 0.31
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
numero_de_elementos = np.sum(condicion_cortical)
print(f"Número de elementos que cumplen con la condición BVTV > 0.29: {numero_de_elementos}")
array_modulo_E_cortical=np.empty_like(array_BVTV)
array_modulo_E2_cortical=np.empty_like(array_BVTV)
array_alpha = np.empty_like(array_BVTV)
array_alpha.fill(0.621594349)
array_vm = np.empty_like(array_BVTV)
array_BVTV = array_densidad_gr_cm_3/(1.41 +1.29*array_alpha)
array_porosidad = 1 - array_BVTV
array_BVTV_previous= np.empty_like(array_BVTV)
array_BVTV_previous.fill(1)

# Calcula el valor promedio de array_densidad
promedio_porosidad = np.mean(array_porosidad)
print("El valor promedio de la porosidad es:", promedio_porosidad)
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------

array_modulo_E_cortical[condicion_cortical] = 84370*(array_BVTV[condicion_cortical] ** 2.58) * (array_alpha[condicion_cortical] ** 2.74)*1e3
mean_array_modulo_E_cortical = 84370*(array_BVTV[condicion_cortical] ** 2.58) * (array_alpha[condicion_cortical] ** 2.74)*1e3
escalar_mean_modulo_cortical= np.mean(mean_array_modulo_E_cortical)
promedio_E_cortical = np.mean(array_modulo_E_cortical)
promedio_E = np.mean(array_modulo_E_cortical)
print("----------------- Evaluación del daño inicial.-----------------")
print("El valor promedio de la E cortical es:", promedio_E_cortical)
print("El valor promedio de la E es:", promedio_E)


#promedio_E_0 = np.mean(array_modulo_E_0)
#print("El valor promedio de la E_0 es:", promedio_E_0)
#array_damage_cortical = 1- (array_modulo_E_cortical/ array_modulo_E_f)

#promedio_daño = np.mean(array_damage_cortical)
#print("El valor promedio de daño inicial es ++++>>>>>:", promedio_daño )


# Definir las constantes de tasa y el ancho (estos valores necesitan ser definidos)
ratecon1 = 0.04
ratecon2 = 0.0
ratecon3 = 0.0
ratecon4 = 0.02
width = 12.5
escalar_km = 5e-4

# Calcular la tasa de aposición lineal basada en el error y almacenar en array_rdot

# --------------------------------------------------------------------------------------------
# aca inicia fase 1 
# --------------------------------------------------------------------------------------------
for dmes in range(1, 2):
    # --------------------------------------------------------------------------------------------
    # aca inicia fase 1 
    # --------------------------------------------------------------------------------------------

    for dday in range(1, 25):  # Ciclo externo con dtime va de 1 a 24
       
        escalar_contador_total_dias += 1
        escalar_contador_total_dias_modelo += 1

        t_prima_24 = 24 - dday
        tmp_day = dday


        for cycle in range(1000, 10001, 1000):  # Ciclo interno va de 1000 en 1000 hasta 10000
            tmp_day = dday
            tmp_cycle = cycle / 1000
            print(f"energy {dday}:{energy} - cycle:{cycle}")

            expression_array_ei = ((2 * energy) / array_modulo_E_f)**(1/2)
            array_psi = (expression_array_ei * cycle)**(1/4)
            fem_function_array_psi.x.array[:]= array_psi
            # ---------------------------------------------------------------------------------------------------------
            # Cálculo del error.
            # ---------------------------------------------------------------------------------------------------------
            array_error= ((1/array_BVTV)**2)*array_psi - 50
            fem_function_array_error.x.array[:]= array_error

            # ---------------------------------------------------------------------------------------------------------
            # Cálculo del balance focal oseo
            # con el valor de balance focal puedo calcular el valor remodelacion trabecular.
            # ---------------------------------------------------------------------------------------------------------
            array_rdot = np.where(array_error > width,
                      ratecon4 * array_error + (ratecon3 - ratecon4) * width,
                      np.where((array_error <= width) & (array_error >= 0),
                               ratecon3 * array_error,
                               np.where((array_error < 0) & (array_error >= -width),
                                        ratecon2 * array_error,
                                        ratecon1 * array_error + (ratecon1 - ratecon2) * width)))
            fem_function_array_rdot.x.array[:]=array_rdot
            
            # ----------------------------------------------------------------------------------------------------------
            # Paso 4: Calcular el valor de la señal inhibitoria.
            # ----------------------------------------------------------------------------------------------------------
           
            # ---------------------------------------------------------------------------------------------------- 
            # El cálculo de la porosidad proviene del calculo del valor de la superficie específica
            array_porosidad = 1 - array_BVTV 
            array_porosity2 = array_porosidad * array_porosidad
            array_porosity3 = array_porosity2 * array_porosidad
            array_porosity4 = array_porosity3 * array_porosidad
            array_porosity5 = array_porosity4 * array_porosidad
            array_Sv = 0.03226 * array_porosidad - 0.09394 * array_porosity2 + 0.13396 * array_porosity3 - 0.10104 * array_porosity4 + 0.02876 * array_porosity5 
            # --------------------------------------------------------------------------------------
            # El calculo de la variacion de la variacion de las BMUs
            array_derivative_NBMU = array_Sv * array_rdot * escalar_V_bmu * escalar_contador_total_dias_modelo

            expression_array_dot_vr = (array_BVTV) * array_derivative_NBMU * escalar_tmp_ar1 * (t_prima_24 /escalar_TR)**2 
            fem_function_vr_dot.x.array[:] = expression_array_dot_vr
          
            array_BVTV[condicion_trabecular] = array_BVTV[condicion_trabecular] + array_derivative_NBMU[condicion_trabecular] / escalar_TR
            array_BVTV[condicion_cortical] = array_BVTV[condicion_cortical] + expression_array_dot_vr[condicion_cortical] / escalar_TR
            #array_BVTV = array_BVTV + expression_array_dot_vr

            
            #---------------------------------------------------------------------------
            array_BVTV_mean=np.mean(array_BVTV[condicion_cortical])
            array_rho_cortical_gr_cm_3 = array_BVTV[condicion_cortical]*(1.41 +1.29*array_alpha[condicion_cortical]) 
            array_rho_tejido = (1.41 +1.29*array_alpha[condicion_cortical]) 
            promedios_rho_tejido[escalar_contador_total_dias ]= np.mean(array_rho_tejido)
            
            array_modulo_E2_cortical = 84370 * (array_BVTV[condicion_cortical] ** 2.58) * (array_alpha[condicion_cortical] ** 2.74) *1e3
            escalar_modulo_E2_cortical = np.mean(array_modulo_E2_cortical)
            # array_modulo_E_f[condicion_cortical]= 84370 * (array_BVTV[condicion_cortical] ** 2.58) * (array_alpha ** 2.74) *1e3
            array_modulo_E_f= 84370 * (array_BVTV ** 2.58) * (array_alpha ** 2.74) *1e3
            
            array_damage = 1-(array_modulo_E_f/array_modulo_E_0)
            
            array_damage_cortical = 1 - (escalar_modulo_E2_cortical/escalar_mean_modulo_cortical)
            # ----------------------------------------------------------------------------------------------------------
            # Los promedios de salida
            # ----------------------------------------------------------------------------------------------------------
            
            promedios_damage[escalar_contador_total_dias]= np.mean(array_damage_cortical)
            promedios_modulo_elasticidad[escalar_contador_total_dias]= escalar_modulo_E2_cortical
            promedios_rho_cortical_gr_cm_3[escalar_contador_total_dias]= np.mean(array_rho_cortical_gr_cm_3)
            promedios_BVTV[escalar_contador_total_dias]=array_BVTV_mean 
            escalar_energy[escalar_contador_total_dias] = energy
            promedios_alpha[escalar_contador_total_dias]=np.mean(array_alpha)

            # ----------------------------------------------------------------------------------------------------------
            # Aca se modifica densidad 
            # ----------------------------------------------------------------------------------------------------------
            #array_densidad[condicion_cortical] = (1.41 + 1.29 * alpha_ini) * array_BVTV[condicion_cortical]
            array_densidad = (1.41 + 1.29 * array_alpha) * array_BVTV

            fem_function_densidad.x.array[:] = array_densidad*1e-6
            fem_function_densidad_gr_cm3.x.array[:] = array_densidad
            fem_function_BVTV.x.array[:] = array_BVTV
            fem_function_modulo_E_f.x.array[:] = array_modulo_E_f
            fem_function_damage.x.array[:] = array_damage
            fem_function_alpha.x.array[:] = array_alpha

            with XDMFFile(mesh.comm, "./20_01_05_24/fase_1/fem_function_array_modulo.xdmf", "w") as file:
                file.write_mesh(mesh)
                file.write_function(fem_function_modulo_E_f , tmp_cycle)    
            with XDMFFile(mesh.comm, "./20_01_05_24/fase_1/fem_function_array_rdot.xdmf", "w") as file:
                file.write_mesh(mesh)
                file.write_function(fem_function_array_rdot , tmp_cycle)    
            with XDMFFile(mesh.comm, "./20_01_05_24/fase_1/fem_function_array_psi.xdmf", "w") as file:
                file.write_mesh(mesh)
                file.write_function(fem_function_array_psi , tmp_cycle)    
            with XDMFFile(mesh.comm, "./20_01_05_24/fase_1/fem_function_array_error.xdmf", "w") as file:
                file.write_mesh(mesh)
                file.write_function(fem_function_array_error , tmp_cycle)    
            with XDMFFile(mesh.comm, "./20_01_05_24/fase_1/fem_function_array_alpha.xdmf", "w") as file:
                file.write_mesh(mesh)
                file.write_function(fem_function_alpha , tmp_cycle)    
            # # --------------------------------------------------------------------------------------------------
            
            # ##---------------------------------------------------------------------------------------------------
            # ##---------------------------------------------------------------------------------------------------

            # Continúa con la parte de resolución de ecuaciones y condiciones de frontera...
            # Al parecer tambien hay un error en el calculo de mu no es + sino -
            mu = fem_function_modulo_E_f / (2.0 * (1.0 + 0.3))
            lame = fem_function_modulo_E_f * 0.3/ ((1.0 + 0.3) * (1.0 - 2.0 * 0.3)) 
            # revisar el valor de lamnda pues al parecer hay un error  
                                    
            f = ufl.as_vector((0, 0, -fem_function_densidad * 9810 ))
            V = functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
            u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
           
            def sigma(v):
                """Return an expression for the stress σ given a displacement field"""
                return 2.0 * mu * ufl.sym(grad(v)) + lame * ufl.tr(ufl.sym(grad(v))) * ufl.Identity(3)

            def strain(u):
                return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
            a = form(inner(sigma(u), grad(v)) * dx)
            L = form(inner(f, v) * dx)

            def invariants_principal(A):
                i1 = ufl.tr(A)
                i2 = (ufl.tr(A)**2 - ufl.tr(A * A)) / 2
                i3 = ufl.det(A)
                return i1, i2, i3

            #----------------------------------------------------------------------------
            # Asignacion de la condición frontera 08/02/24 - R-ninety
            #----------------------------------------------------------------------------
            def boundary_z(x):
                return np.logical_and(x[2] >= 36, x[2] <= 37)
            
            # Ubicar los grados de libertad en las caras con z entre 36 y 37
            dofs = dolfinx.fem.locate_dofs_geometrical(V, boundary_z)
            
            # Definir la condición de frontera como desplazamiento 0 en todas las direcciones
            u_D = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0, 0, 0)))
            
            # Aplicar la condición de frontera
            bc = dolfinx.fem.dirichletbc(u_D, dofs, V)
            #----------------------------------------------------------------------------
            
            # ## Assemble and solve
            #
            # The bilinear form `a` is assembled into a matrix `A`, with
            # modifications for the Dirichlet boundary conditions. The call
            # `A.assemble()` completes any parallel communication required to
            # compute the matrix.


            A = assemble_matrix(a, bcs=[bc])
            A.assemble()
            
            # Coordenadas de los puntos donde se aplicarán las cargas puntuales
            point_138 = np.array([87.7431, 20.6717, 224.6261],dtype=np.float64)
            point_140 = np.array([85.6372, 33.9296, 226.9851],dtype=np.float64)
            point_141 = np.array([77.7509, 23.4239, 216.3084],dtype=np.float64)
            point_143 = np.array([78.43, 17.2494, 209.5734],dtype=np.float64)
            point_148 = np.array([73.5285, 30.5294, 203.1597],dtype=np.float64)
            point_52 = np.array([124.2809, 54.4087,180.5598],dtype=np.float64)

            # Cargas en la dirección Z y X

            def marker_52(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_52[0], atol=atol)
                close_y = np.isclose(x[1], point_52[1], atol=atol)
                close_z = np.isclose(x[2], point_52[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_138(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_138[0], atol=atol)
                close_y = np.isclose(x[1], point_138[1], atol=atol)
                close_z = np.isclose(x[2], point_138[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_140(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_140[0], atol=atol)
                close_y = np.isclose(x[1], point_140[1], atol=atol)
                close_z = np.isclose(x[2], point_140[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)
            def marker_141(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_141[0], atol=atol)
                close_y = np.isclose(x[1], point_141[1], atol=atol)
                close_z = np.isclose(x[2], point_141[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)
            def marker_143(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_143[0], atol=atol)
                close_y = np.isclose(x[1], point_143[1], atol=atol)
                close_z = np.isclose(x[2], point_143[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_148(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_148[0], atol=atol)
                close_y = np.isclose(x[1], point_148[1], atol=atol)
                close_z = np.isclose(x[2], point_148[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)



            b = assemble_vector(L)

            # Localiza los grados de libertad en el vértice
            vertices_52 = locate_dofs_geometrical(V, marker_138)
            dofs_52_x = (vertices_52[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_52_z = (vertices_52[0])*3 + 2  # Grado de libertad en la dirección z


            vertices_138 = locate_dofs_geometrical(V, marker_138)
            dofs_138_x = (vertices_138[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_138_z = (vertices_138[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_140 = locate_dofs_geometrical(V, marker_140)
            dofs_140_x = (vertices_140[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_140_z = (vertices_140[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_141 = locate_dofs_geometrical(V, marker_141)
            dofs_141_x = (vertices_141[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_141_z = (vertices_141[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_143 = locate_dofs_geometrical(V, marker_143)
            dofs_143_x = (vertices_143[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_143_z = (vertices_143[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_148 = locate_dofs_geometrical(V, marker_148)
            dofs_148_x = (vertices_148[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_148_z = (vertices_148[0])*3 + 2  # Grado de libertad en la dirección z



            #print("El valor de los DOFs en X y en Z en la  carga:", dofs_138_x, dofs_138_z)
            # Magnitudes de las cargas puntuales
            P_x = 0  # Carga en la dirección y 
            P_z = -2317   # Carga en la dirección z
            P_z2= 713       # Carga del abductor en z
                # Aplicar las cargas puntuales
            with b.localForm() as loc_b:
                loc_b.setValues(dofs_138_x, P_x)
                loc_b.setValues(dofs_138_z, P_z)
                loc_b.setValues(dofs_140_x, P_x)
                loc_b.setValues(dofs_140_z, P_z) 
                loc_b.setValues(dofs_141_x, P_x)
                loc_b.setValues(dofs_141_z, P_z) 
                loc_b.setValues(dofs_143_x, P_x)
                loc_b.setValues(dofs_143_z, P_z) 
                loc_b.setValues(dofs_148_x, P_x)
                loc_b.setValues(dofs_148_z, P_z)
                loc_b.setValues(dofs_52_x, P_x)
                loc_b.setValues(dofs_52_z, P_z2)

            
            apply_lifting(b, [a], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
            set_bc(b, [bc])
    
    
            ns = build_nullspace(V)
            A.setNearNullSpace(ns)
            A.setOption(PETSc.Mat.Option.SPD, True)  # type: ignore
    
            
            # +
            # Set solver options
            opts = PETSc.Options()  # type: ignore
            opts["ksp_type"] = "cg"
            opts["ksp_rtol"] = 1.0e-8
            opts["pc_type"] = "gamg"
        
            # Use Chebyshev smoothing for multigrid
            opts["mg_levels_ksp_type"] = "chebyshev"
            opts["mg_levels_pc_type"] = "jacobi"
        
            # Improve estimate of eigenvalues for Chebyshev smoothing
            opts["mg_levels_ksp_chebyshev_esteig_steps"] = 10
        
            # Create PETSc Krylov solver and turn convergence monitoring on
            solver = PETSc.KSP().create(mesh.comm)  # type: ignore
            solver.setFromOptions()
    
            # Set matrix operator
            solver.setOperators(A)
            
            # +
            uh2_in = Function(V)
    
            solver.solve(b, uh2_in.vector)
    
            uh2_in.x.scatter_forward()
            
            # ## Post-processing
            # +
            # sigma_dev = sigma(uh_in) - (1 / 3) * ufl.tr(sigma(uh_in)) * ufl.Identity(len(uh_in))
            # sigma_vm = ufl.sqrt((3 / 2) * inner(sigma_dev, sigma_dev))
            # W = functionspace(mesh, ("Discontinuous Lagrange", 0))
            
            # sigma_vm_expr = Expression(sigma_vm, W.element.interpolation_points())
            # sigma_vm_h = Function(W,name="Von_misses")
            # sigma_vm_h.interpolate(sigma_vm_expr)

            
            # ----------------------------------------------------------------------------------

            
           
            U = 0.5 * ufl.inner(sigma(uh2_in), strain(uh2_in)) * ufl.dx
    
            # Ensamblar la energía de deformación ----<< 09/02/24 ---<<< OKß
            energy = fem.assemble_scalar(fem.form(U)) 
           # print("Energía de deformación total:", energy)
           # print("Sigma_bar:", sigma_bar_i)
            
            
            
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_1/fem_function_densidad_fase_1.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_densidad_gr_cm3 , tmp_day)
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_1/fem_function_BVTV_fase_1.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_BVTV , tmp_day)
    print(f"El valor promedio de daño:{promedios_damage[escalar_contador_total_dias]}")
    print(f"El valor promedio de la densidad:{promedios_rho_cortical_gr_cm_3[escalar_contador_total_dias]}")
    print(f"El valor promedio de la fracción de volumen:{promedios_BVTV[escalar_contador_total_dias]}")
    print(f"1.{dday} - El valor promedio del modulo de elasticidad:{promedios_modulo_elasticidad[escalar_contador_total_dias]}")
    print(f" Finalizo la primera etapa -->> La energía: {escalar_energy[escalar_contador_total_dias]}")
  
# ----------------------------------------------------------------------------------------------------
# aca inicia fase 2
# ----------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
# Para fase #2  el tiempo es menor de 8 días
# expression_array_dot_vr = np.zeros(W.dofmap.index_map.size_local)
# Phase 2            # Only  resorption aca deberia solamente  r hasta el día 32---->>>>>
#   for TR in range(1, 1, 24):  # Ciclo interno va de 1 en 1 hasta 24 
#---------------------------------------------------------------------------------------------------
      
    promedio_BVTV= np.mean(array_BVTV)
    print("---------------------------------------------------------")              
    print("El valor promedio de la BVTV es:", promedios_BVTV[escalar_contador_total_dias])
    print(f" Inicia la segunda etapa -->> La energía: {energy}")
    print("---------------------------------------------------------")
    tmp_lenght = escalar_contador_total_dias
    print(f"01-longitud del CSV-->>>>:{tmp_lenght}")
    for dday in range(1, 8):  # Ciclo externo con dtime va de 1 a 8
        
        tmp_day = dday
        escalar_contador_total_dias += 1
        escalar_contador_total_dias_modelo += 1
        t_prima_8 = 8 - dday
    
        for cycle in range(1000, 10001, 1000):  # Ciclo interno va de 1000 en 1000 hasta 10000
            tmp_cycle = cycle / 1000
    
            expression_array_ei = ((2 * energy) / array_modulo_E_f) ** (1 / 2)
            array_psi = (expression_array_ei * cycle) ** (1 / 4)
            fem_function_array_psi.x.array[:]= array_psi

            # ---------------------------------------------------------------------------------------------------------
            # Cálculo del error.
            # ---------------------------------------------------------------------------------------------------------
            array_error= ((1/array_BVTV)**2)*array_psi - 50
            fem_function_array_error.x.array[:]= array_error

            # ---------------------------------------------------------------------------------------------------------
            # Cálculo del balance focal oseo
            # con el valor de balance focal puedo calcular el valor remodelacion trabecular.
            # ---------------------------------------------------------------------------------------------------------
            array_rdot = np.where(array_error > width,
                      ratecon4 * array_error + (ratecon3 - ratecon4) * width,
                      np.where((array_error <= width) & (array_error >= 0),
                               ratecon3 * array_error,
                               np.where((array_error < 0) & (array_error >= -width),
                                        ratecon2 * array_error,
                                        ratecon1 * array_error + (ratecon1 - ratecon2) * width)))
            fem_function_array_rdot.x.array[:]=array_rdot
            

            array_porosidad = 1 - array_BVTV
            array_porosity2 = array_porosidad * array_porosidad
            array_porosity3 = array_porosity2 * array_porosidad
            array_porosity4 = array_porosity3 * array_porosidad
            array_porosity5 = array_porosity4 * array_porosidad
            array_Sv = 0.03226 * array_porosidad - 0.09394 * array_porosity2 + 0.13396 * array_porosity3 - 0.10104 * array_porosity4 + 0.02876 * array_porosity5
    
            array_derivative_NBMU = array_Sv * array_rdot *escalar_V_bmu * escalar_contador_total_dias_modelo
            # ------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------
            # Aca se esta modificando la fraccion de volumen de hueso cortical                        
            # ------------------------------------------------------------------------------------------------

            expression_array_dot_vr = array_BVTV * array_derivative_NBMU * escalar_tmp_ar1 / escalar_TR
            fem_function_vr_dot.x.array[:] = expression_array_dot_vr
    
            array_BVTV[condicion_trabecular] = array_BVTV[condicion_trabecular] - array_derivative_NBMU[condicion_trabecular] /escalar_TR
            array_BVTV[condicion_cortical] = array_BVTV[condicion_cortical] + expression_array_dot_vr[condicion_cortical] #+ expression_array_dot_vr[condicion_cortical] * 0.00034 / array_BVTV[condicion_cortical]
            #-------------------------------------------------------------------------------------------------------
            array_BVTV_mean=np.mean(array_BVTV[condicion_cortical])
            array_rho_cortical_gr_cm_3 = array_BVTV[condicion_cortical]*(1.41 +1.29*array_alpha[condicion_cortical]) 
            array_rho_tejido = (1.41 +1.29*array_alpha[condicion_cortical]) 
            promedios_rho_tejido[escalar_contador_total_dias]= np.mean(array_rho_tejido)
            
            
            array_modulo_E2_cortical = 84370 * (array_BVTV[condicion_cortical] ** 2.58) * (array_alpha[condicion_cortical] ** 2.74) *1e3
            escalar_modulo_E2_cortical = np.mean(array_modulo_E2_cortical)
            # array_modulo_E_f[condicion_cortical]= 84370 * (array_BVTV[condicion_cortical] ** 2.58) * (array_alpha ** 2.74) *1e3
            array_modulo_E_f = 84370 * (array_BVTV ** 2.58) * (array_alpha ** 2.74) *1e3
            
            array_damage = 1-(array_modulo_E_f/array_modulo_E_0)
            array_damage_cortical = 1 - (escalar_modulo_E2_cortical/escalar_mean_modulo_cortical)
            
            #-------------------------------------------------------------------------------------------------------            
            promedios_damage[escalar_contador_total_dias]= np.mean(array_damage_cortical)
            promedios_modulo_elasticidad[escalar_contador_total_dias]= escalar_modulo_E2_cortical
            promedios_rho_cortical_gr_cm_3[escalar_contador_total_dias]= np.mean(array_rho_cortical_gr_cm_3)
            promedios_BVTV[escalar_contador_total_dias]=array_BVTV_mean 
            escalar_energy[escalar_contador_total_dias] = energy
            promedios_alpha[escalar_contador_total_dias]=np.mean(array_alpha)


    
            
            # ----------------------------------------------------------------------------------------------------------
            # Aca se modifica densidad 
            # ----------------------------------------------------------------------------------------------------------
            #array_densidad[condicion_cortical] = (1.41 + 1.29 * alpha_ini) * array_BVTV[condicion_cortical]
            array_densidad = (1.41 + 1.29 * alpha_ini) * array_BVTV
            fem_function_densidad_gr_cm3.x.array[:] = array_densidad
            fem_function_densidad.x.array[:] = array_densidad*1e-6
            fem_function_BVTV.x.array[:] = array_BVTV
            fem_function_modulo_E_f.x.array[:] = array_modulo_E_f
            fem_function_damage.x.array[:] = array_damage
            
            # # ----------------------------------------------------------------------------------------------------------
            with XDMFFile(mesh.comm, "./20_01_05_24/fase_2/fem_function_array_modulo.xdmf", "w") as file:
                file.write_mesh(mesh)
                file.write_function(fem_function_modulo_E_f , tmp_cycle)    
            with XDMFFile(mesh.comm, "./20_01_05_24/fase_2/fem_function_array_damage.xdmf", "w") as file:
                file.write_mesh(mesh)
                file.write_function(fem_function_damage , tmp_cycle)    
            with XDMFFile(mesh.comm, "./20_01_05_24/fase_2/fem_function_array_rdot.xdmf", "w") as file:
                file.write_mesh(mesh)
                file.write_function(fem_function_array_rdot , tmp_cycle)    
            with XDMFFile(mesh.comm, "./20_01_05_24/fase_2/fem_function_array_psi.xdmf", "w") as file:
                file.write_mesh(mesh)
                file.write_function(fem_function_array_psi , tmp_cycle)    
            with XDMFFile(mesh.comm, "./20_01_05_24/fase_2/fem_function_array_error.xdmf", "w") as file:
                file.write_mesh(mesh)
                file.write_function(fem_function_array_error , tmp_cycle)    
    
            # ##---------------------------------------------------------------------------------------------------
            # ##---------------------------------------------------------------------------------------------------

            # Continúa con la parte de resolución de ecuaciones y condiciones de frontera...
            # Al parecer tambien hay un error en el calculo de mu no es + sino -
            mu = fem_function_modulo_E_f / (2.0 * (1.0 + 0.3))
            lame = fem_function_modulo_E_f * 0.3/ ((1.0 + 0.3) * (1.0 - 2.0 * 0.3)) 
            # revisar el valor de lamnda pues al parecer hay un error  
                                    
            f = ufl.as_vector((0, 0, -fem_function_densidad * 9810 ))
            V = functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
            u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
           
            def sigma(v):
                """Return an expression for the stress σ given a displacement field"""
                return 2.0 * mu * ufl.sym(grad(v)) + lame * ufl.tr(ufl.sym(grad(v))) * ufl.Identity(3)

            def strain(u):
                return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
            a = form(inner(sigma(u), grad(v)) * dx)
            L = form(inner(f, v) * dx)

            def invariants_principal(A):
                i1 = ufl.tr(A)
                i2 = (ufl.tr(A)**2 - ufl.tr(A * A)) / 2
                i3 = ufl.det(A)
                return i1, i2, i3

            #----------------------------------------------------------------------------
            # Asignacion de la condición frontera 08/02/24 - R-ninety
            #----------------------------------------------------------------------------
            def boundary_z(x):
                return np.logical_and(x[2] >= 36, x[2] <= 37)
            
            # Ubicar los grados de libertad en las caras con z entre 36 y 37
            dofs = dolfinx.fem.locate_dofs_geometrical(V, boundary_z)
            
            # Definir la condición de frontera como desplazamiento 0 en todas las direcciones
            u_D = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0, 0, 0)))
            
            # Aplicar la condición de frontera
            bc = dolfinx.fem.dirichletbc(u_D, dofs, V)
            #----------------------------------------------------------------------------
            
            # ## Assemble and solve
            #
            # The bilinear form `a` is assembled into a matrix `A`, with
            # modifications for the Dirichlet boundary conditions. The call
            # `A.assemble()` completes any parallel communication required to
            # compute the matrix.


            A = assemble_matrix(a, bcs=[bc])
            A.assemble()
            
            

            # Coordenadas de los puntos donde se aplicarán las cargas puntuales
            point_138 = np.array([87.7431, 20.6717, 224.6261],dtype=np.float64)
            point_140 = np.array([85.6372, 33.9296, 226.9851],dtype=np.float64)
            point_141 = np.array([77.7509, 23.4239, 216.3084],dtype=np.float64)
            point_143 = np.array([78.43, 17.2494, 209.5734],dtype=np.float64)
            point_148 = np.array([73.5285, 30.5294, 203.1597],dtype=np.float64)
            point_52 = np.array([124.2809, 54.4087,180.5598],dtype=np.float64)

            # Cargas en la dirección Z y X

            def marker_52(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_52[0], atol=atol)
                close_y = np.isclose(x[1], point_52[1], atol=atol)
                close_z = np.isclose(x[2], point_52[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_138(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_138[0], atol=atol)
                close_y = np.isclose(x[1], point_138[1], atol=atol)
                close_z = np.isclose(x[2], point_138[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_140(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_140[0], atol=atol)
                close_y = np.isclose(x[1], point_140[1], atol=atol)
                close_z = np.isclose(x[2], point_140[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)
            def marker_141(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_141[0], atol=atol)
                close_y = np.isclose(x[1], point_141[1], atol=atol)
                close_z = np.isclose(x[2], point_141[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)
            def marker_143(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_143[0], atol=atol)
                close_y = np.isclose(x[1], point_143[1], atol=atol)
                close_z = np.isclose(x[2], point_143[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_148(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_148[0], atol=atol)
                close_y = np.isclose(x[1], point_148[1], atol=atol)
                close_z = np.isclose(x[2], point_148[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)



            b = assemble_vector(L)

            # Localiza los grados de libertad en el vértice
            vertices_52 = locate_dofs_geometrical(V, marker_138)
            dofs_52_x = (vertices_52[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_52_z = (vertices_52[0])*3 + 2  # Grado de libertad en la dirección z


            vertices_138 = locate_dofs_geometrical(V, marker_138)
            dofs_138_x = (vertices_138[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_138_z = (vertices_138[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_140 = locate_dofs_geometrical(V, marker_140)
            dofs_140_x = (vertices_140[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_140_z = (vertices_140[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_141 = locate_dofs_geometrical(V, marker_141)
            dofs_141_x = (vertices_141[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_141_z = (vertices_141[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_143 = locate_dofs_geometrical(V, marker_143)
            dofs_143_x = (vertices_143[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_143_z = (vertices_143[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_148 = locate_dofs_geometrical(V, marker_148)
            dofs_148_x = (vertices_148[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_148_z = (vertices_148[0])*3 + 2  # Grado de libertad en la dirección z



            #print("El valor de los DOFs en X y en Z en la  carga:", dofs_138_x, dofs_138_z)
            # Magnitudes de las cargas puntuales
            P_x = 0  # Carga en la dirección y 
            P_z = -2317   # Carga en la dirección z
            P_z2= 713       # Carga del abductor en z
                # Aplicar las cargas puntuales
            with b.localForm() as loc_b:
                loc_b.setValues(dofs_138_x, P_x)
                loc_b.setValues(dofs_138_z, P_z)
                loc_b.setValues(dofs_140_x, P_x)
                loc_b.setValues(dofs_140_z, P_z) 
                loc_b.setValues(dofs_141_x, P_x)
                loc_b.setValues(dofs_141_z, P_z) 
                loc_b.setValues(dofs_143_x, P_x)
                loc_b.setValues(dofs_143_z, P_z) 
                loc_b.setValues(dofs_148_x, P_x)
                loc_b.setValues(dofs_148_z, P_z)
                loc_b.setValues(dofs_52_x, P_x)
                loc_b.setValues(dofs_52_z, P_z2)
            
            
            apply_lifting(b, [a], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
            set_bc(b, [bc])
    
    
            ns = build_nullspace(V)
            A.setNearNullSpace(ns)
            A.setOption(PETSc.Mat.Option.SPD, True)  # type: ignore
    
            
            # +
            # Set solver options
            opts = PETSc.Options()  # type: ignore
            opts["ksp_type"] = "cg"
            opts["ksp_rtol"] = 1.0e-8
            opts["pc_type"] = "gamg"
        
            # Use Chebyshev smoothing for multigrid
            opts["mg_levels_ksp_type"] = "chebyshev"
            opts["mg_levels_pc_type"] = "jacobi"
        
            # Improve estimate of eigenvalues for Chebyshev smoothing
            opts["mg_levels_ksp_chebyshev_esteig_steps"] = 10
        
            # Create PETSc Krylov solver and turn convergence monitoring on
            solver = PETSc.KSP().create(mesh.comm)  # type: ignore
            solver.setFromOptions()
    
            # Set matrix operator
            solver.setOperators(A)
            
            # +
            uh2_in = Function(V)
    
            solver.solve(b, uh2_in.vector)
    
            uh2_in.x.scatter_forward()
            
            # ## Post-processing
            # +
            # sigma_dev = sigma(uh_in) - (1 / 3) * ufl.tr(sigma(uh_in)) * ufl.Identity(len(uh_in))
            # sigma_vm = ufl.sqrt((3 / 2) * inner(sigma_dev, sigma_dev))
            # W = functionspace(mesh, ("Discontinuous Lagrange", 0))
            
            # sigma_vm_expr = Expression(sigma_vm, W.element.interpolation_points())
            # sigma_vm_h = Function(W,name="Von_misses")
            # sigma_vm_h.interpolate(sigma_vm_expr)

            
            # ----------------------------------------------------------------------------------

            
           
            U = 0.5 * ufl.inner(sigma(uh2_in), strain(uh2_in)) * ufl.dx
    
            # Ensamblar la energía de deformación ----<< 09/02/24 ---<<< OKß
            energy = fem.assemble_scalar(fem.form(U)) 
           # print("Energía de deformación total:", energy)
           # print("Sigma_bar:", sigma_bar_i)
            print(f"El valor promedio de daño:{promedios_damage[escalar_contador_total_dias]}")
            print(f"El valor promedio de la densidad:{promedios_rho_cortical_gr_cm_3[escalar_contador_total_dias]}")
            print(f"El valor promedio de la fracción de volumen:{promedios_BVTV[escalar_contador_total_dias]}")
            print(f"2.{dday} - El valor promedio del modulo de elasticidad:{promedios_modulo_elasticidad[escalar_contador_total_dias]}")

    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_2/fem_function_densidad.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_densidad_gr_cm3 , tmp_day)
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_2/fem_function_BVTV.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_BVTV , tmp_day)

    print("---------------------------------------------------------")              
    print("El valor promedio de la BVTV es:", promedios_BVTV[escalar_contador_total_dias])
    print(f" Inicia la tercera etapa -->> La energía: {energy}")
    print("---------------------------------------------------------")
    tmp_lenght = escalar_contador_total_dias
    print(f"02-longitud del CSV-->>>>:{tmp_lenght}")  
   
   
#----------------------------------------------------------------------------------------------
# Inicia fase 3 Desarrollo del frente de formación
# 
#----------------------------------------------------------------------------------------------
    for dday in range(1, 64):  # Ciclo externo con dtime va de 1 a 64
  
        escalar_contador_total_dias += 1
        escalar_contador_total_dias_modelo += 1
        t_prima_64 = 64 - dday
        tmp_day = dday

        for cycle in range(1000, 10001, 1000):  # Ciclo interno va de 1000 en 1000 hasta 10000
            tmp_day = dday
            tmp_cycle = cycle / 1000
    
            expression_array_ei = ((2 * energy) / array_modulo_E_f) ** (1 / 2)
            array_psi = (expression_array_ei * cycle) ** (1 / 4)
            fem_function_array_psi.x.array[:]= array_psi

     
            # ---------------------------------------------------------------------------------------------------------
            # Cálculo del error.
            # ---------------------------------------------------------------------------------------------------------
            array_error= ((1/array_BVTV)**2)*array_psi - 50
            fem_function_array_error.x.array[:]= array_error

            # ---------------------------------------------------------------------------------------------------------
            # Cálculo del balance focal oseo
            # con el valor de balance focal puedo calcular el valor remodelacion trabecular.
            # ---------------------------------------------------------------------------------------------------------
            array_rdot = np.where(array_error > width,
                      ratecon4 * array_error + (ratecon3 - ratecon4) * width,
                      np.where((array_error <= width) & (array_error >= 0),
                               ratecon3 * array_error,
                               np.where((array_error < 0) & (array_error >= -width),
                                        ratecon2 * array_error,
                                        ratecon1 * array_error + (ratecon1 - ratecon2) * width)))
            fem_function_array_rdot.x.array[:]=array_rdot


            expression_array_s = (array_psi / (1 + array_psi)) * (1 - array_damage) ** (1)
            fem_function_array_s = fem.Function(W)
            
            fem_function_array_s.x.array[:] = expression_array_s
    
            array_for = escalar_fbio * (1 - expression_array_s)
    
            array_porosidad = 1 - array_BVTV
            array_porosity2 = array_porosidad * array_porosidad
            array_porosity3 = array_porosity2 * array_porosidad
            array_porosity4 = array_porosity3 * array_porosidad
            array_porosity5 = array_porosity4 * array_porosidad
            array_Sv = 0.03226 * array_porosidad - 0.09394 * array_porosity2 + 0.13396 * array_porosity3 - 0.10104 * array_porosity4 + 0.02876 * array_porosity5
    
            array_derivative_NBMU = array_Sv * array_rdot *escalar_V_bmu * escalar_contador_total_dias_modelo
            
            # escalar_tmp_ar2 = 0.004536  # Este valor debe de estar en days/mm[2] Valor para fase Formacion fase 3 
            # TR = 24
            # TI= 8
            # TF= 64
            # t= 1
            # d_o  = 0,076
            # d_H= 0,0145
            # d_E = 0,0491
            # d_BMU= 0,152
            escalar_f1 = 0.1908 * (t_prima_64) / escalar_TF
            escalar_tmp_af3 = escalar_tmp_ar2 * (1 - escalar_f1 ** 2)
            
            
            expression_array_dot_vb = array_derivative_NBMU * escalar_tmp_af3 /escalar_TF
            fem_function_vb_dot.x.array[:] = expression_array_dot_vb
            
            array_formacion_vb[condicion_trabecular]= array_formacion_vb[condicion_trabecular] + array_derivative_NBMU[condicion_trabecular]/escalar_TF
            array_formacion_vb[condicion_cortical] = array_formacion_vb[condicion_cortical] + expression_array_dot_vb[condicion_cortical]
    
    
    
            array_BVTV[condicion_trabecular] = array_BVTV[condicion_trabecular] + array_derivative_NBMU[condicion_trabecular]/escalar_TF
            array_BVTV[condicion_cortical] = array_BVTV[condicion_cortical] - expression_array_dot_vb[condicion_cortical]
            #-------------------------------------------------------------------------------------------------------
            array_BVTV_mean=np.mean(array_BVTV[condicion_cortical])
            array_rho_cortical_gr_cm_3 = array_BVTV[condicion_cortical]*(1.41 +1.29*array_alpha[condicion_cortical]) 
            array_rho_tejido = (1.41 +1.29*array_alpha[condicion_cortical]) 
            promedios_rho_tejido[escalar_contador_total_dias]= np.mean(array_rho_tejido)
            
            
            array_modulo_E2_cortical = 84370 * (array_BVTV[condicion_cortical] ** 2.58) * (array_alpha[condicion_cortical] ** 2.74) *1e3
            escalar_modulo_E2_cortical = np.mean(array_modulo_E2_cortical)
            #array_modulo_E_f[condicion_cortical]= 84370 * (array_BVTV[condicion_cortical] ** 2.58) * (array_alpha ** 2.74) *1e3
            array_modulo_E_f = 84370 * (array_BVTV ** 2.58) * (array_alpha ** 2.74) *1e3

            array_damage = 1-(array_modulo_E_f/array_modulo_E_0)
            array_damage_cortical = 1 - (escalar_modulo_E2_cortical/escalar_mean_modulo_cortical)
            
            
            promedios_damage[escalar_contador_total_dias]= np.mean(array_damage_cortical)
            promedios_modulo_elasticidad[escalar_contador_total_dias]= escalar_modulo_E2_cortical
            promedios_rho_cortical_gr_cm_3[escalar_contador_total_dias]= np.mean(array_rho_cortical_gr_cm_3)
            promedios_BVTV[escalar_contador_total_dias]=array_BVTV_mean 
            escalar_energy[escalar_contador_total_dias] = energy
            promedios_alpha[escalar_contador_total_dias]=np.mean(array_alpha)



            
            # ----------------------------------------------------------------------------------------------------------
            # Aca se modifica densidad 
            # ----------------------------------------------------------------------------------------------------------
            #array_densidad[condicion_cortical] = (1.41 + 1.29 * alpha_ini) * array_BVTV[condicion_cortical]
            array_densidad = (1.41 + 1.29 * array_alpha) * array_BVTV
            fem_function_densidad.x.array[:] = array_densidad*1e-6
            fem_function_densidad_gr_cm3.x.array[:] = array_densidad
            fem_function_BVTV.x.array[:] = array_BVTV
            fem_function_modulo_E_f.x.array[:] = array_modulo_E_f
            fem_function_damage.x.array[:] = array_damage
            
            # ##---------------------------------------------------------------------------------------------------
            # ##---------------------------------------------------------------------------------------------------

            # Continúa con la parte de resolución de ecuaciones y condiciones de frontera...
            # Al parecer tambien hay un error en el calculo de mu no es + sino -
            mu = fem_function_modulo_E_f / (2.0 * (1.0 + 0.3))
            lame = fem_function_modulo_E_f * 0.3/ ((1.0 + 0.3) * (1.0 - 2.0 * 0.3)) 
            # revisar el valor de lamnda pues al parecer hay un error  
                                    
            f = ufl.as_vector((0, 0, -fem_function_densidad * 9810 ))
            V = functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
            u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
           
            def sigma(v):
                """Return an expression for the stress σ given a displacement field"""
                return 2.0 * mu * ufl.sym(grad(v)) + lame * ufl.tr(ufl.sym(grad(v))) * ufl.Identity(3)

            def strain(u):
                return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
            a = form(inner(sigma(u), grad(v)) * dx)
            L = form(inner(f, v) * dx)

            def invariants_principal(A):
                i1 = ufl.tr(A)
                i2 = (ufl.tr(A)**2 - ufl.tr(A * A)) / 2
                i3 = ufl.det(A)
                return i1, i2, i3

            #----------------------------------------------------------------------------
            # Asignacion de la condición frontera 08/02/24 - R-ninety
            #----------------------------------------------------------------------------
            def boundary_z(x):
                return np.logical_and(x[2] >= 36, x[2] <= 37)
            
            # Ubicar los grados de libertad en las caras con z entre 36 y 37
            dofs = dolfinx.fem.locate_dofs_geometrical(V, boundary_z)
            
            # Definir la condición de frontera como desplazamiento 0 en todas las direcciones
            u_D = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0, 0, 0)))
            
            # Aplicar la condición de frontera
            bc = dolfinx.fem.dirichletbc(u_D, dofs, V)
            #----------------------------------------------------------------------------
            
            # ## Assemble and solve
            #
            # The bilinear form `a` is assembled into a matrix `A`, with
            # modifications for the Dirichlet boundary conditions. The call
            # `A.assemble()` completes any parallel communication required to
            # compute the matrix.


            A = assemble_matrix(a, bcs=[bc])
            A.assemble()
            
            

            # Coordenadas de los puntos donde se aplicarán las cargas puntuales
            point_138 = np.array([87.7431, 20.6717, 224.6261],dtype=np.float64)
            point_140 = np.array([85.6372, 33.9296, 226.9851],dtype=np.float64)
            point_141 = np.array([77.7509, 23.4239, 216.3084],dtype=np.float64)
            point_143 = np.array([78.43, 17.2494, 209.5734],dtype=np.float64)
            point_148 = np.array([73.5285, 30.5294, 203.1597],dtype=np.float64)
            point_52 = np.array([124.2809, 54.4087,180.5598],dtype=np.float64)

            # Cargas en la dirección Z y X

            def marker_52(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_52[0], atol=atol)
                close_y = np.isclose(x[1], point_52[1], atol=atol)
                close_z = np.isclose(x[2], point_52[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_138(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_138[0], atol=atol)
                close_y = np.isclose(x[1], point_138[1], atol=atol)
                close_z = np.isclose(x[2], point_138[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_140(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_140[0], atol=atol)
                close_y = np.isclose(x[1], point_140[1], atol=atol)
                close_z = np.isclose(x[2], point_140[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)
            def marker_141(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_141[0], atol=atol)
                close_y = np.isclose(x[1], point_141[1], atol=atol)
                close_z = np.isclose(x[2], point_141[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)
            def marker_143(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_143[0], atol=atol)
                close_y = np.isclose(x[1], point_143[1], atol=atol)
                close_z = np.isclose(x[2], point_143[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_148(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_148[0], atol=atol)
                close_y = np.isclose(x[1], point_148[1], atol=atol)
                close_z = np.isclose(x[2], point_148[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)



            b = assemble_vector(L)

            # Localiza los grados de libertad en el vértice
            vertices_52 = locate_dofs_geometrical(V, marker_138)
            dofs_52_x = (vertices_52[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_52_z = (vertices_52[0])*3 + 2  # Grado de libertad en la dirección z


            vertices_138 = locate_dofs_geometrical(V, marker_138)
            dofs_138_x = (vertices_138[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_138_z = (vertices_138[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_140 = locate_dofs_geometrical(V, marker_140)
            dofs_140_x = (vertices_140[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_140_z = (vertices_140[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_141 = locate_dofs_geometrical(V, marker_141)
            dofs_141_x = (vertices_141[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_141_z = (vertices_141[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_143 = locate_dofs_geometrical(V, marker_143)
            dofs_143_x = (vertices_143[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_143_z = (vertices_143[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_148 = locate_dofs_geometrical(V, marker_148)
            dofs_148_x = (vertices_148[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_148_z = (vertices_148[0])*3 + 2  # Grado de libertad en la dirección z



            #print("El valor de los DOFs en X y en Z en la  carga:", dofs_138_x, dofs_138_z)
            # Magnitudes de las cargas puntuales
            P_x = 0  # Carga en la dirección y 
            P_z = -2317   # Carga en la dirección z
            P_z2= 713       # Carga del abductor en z
                # Aplicar las cargas puntuales
            with b.localForm() as loc_b:
                loc_b.setValues(dofs_138_x, P_x)
                loc_b.setValues(dofs_138_z, P_z)
                loc_b.setValues(dofs_140_x, P_x)
                loc_b.setValues(dofs_140_z, P_z) 
                loc_b.setValues(dofs_141_x, P_x)
                loc_b.setValues(dofs_141_z, P_z) 
                loc_b.setValues(dofs_143_x, P_x)
                loc_b.setValues(dofs_143_z, P_z) 
                loc_b.setValues(dofs_148_x, P_x)
                loc_b.setValues(dofs_148_z, P_z)
                loc_b.setValues(dofs_52_x, P_x)
                loc_b.setValues(dofs_52_z, P_z2)
            
            
            apply_lifting(b, [a], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
            set_bc(b, [bc])
    
    
            ns = build_nullspace(V)
            A.setNearNullSpace(ns)
            A.setOption(PETSc.Mat.Option.SPD, True)  # type: ignore
    
            
            # +
            # Set solver options
            opts = PETSc.Options()  # type: ignore
            opts["ksp_type"] = "cg"
            opts["ksp_rtol"] = 1.0e-8
            opts["pc_type"] = "gamg"
        
            # Use Chebyshev smoothing for multigrid
            opts["mg_levels_ksp_type"] = "chebyshev"
            opts["mg_levels_pc_type"] = "jacobi"
        
            # Improve estimate of eigenvalues for Chebyshev smoothing
            opts["mg_levels_ksp_chebyshev_esteig_steps"] = 10
        
            # Create PETSc Krylov solver and turn convergence monitoring on
            solver = PETSc.KSP().create(mesh.comm)  # type: ignore
            solver.setFromOptions()
    
            # Set matrix operator
            solver.setOperators(A)
            
            # +
            uh2_in = Function(V)
    
            solver.solve(b, uh2_in.vector)
    
            uh2_in.x.scatter_forward()
            
            # ## Post-processing
            # +
            # sigma_dev = sigma(uh_in) - (1 / 3) * ufl.tr(sigma(uh_in)) * ufl.Identity(len(uh_in))
            # sigma_vm = ufl.sqrt((3 / 2) * inner(sigma_dev, sigma_dev))
            # W = functionspace(mesh, ("Discontinuous Lagrange", 0))
            
            # sigma_vm_expr = Expression(sigma_vm, W.element.interpolation_points())
            # sigma_vm_h = Function(W,name="Von_misses")
            # sigma_vm_h.interpolate(sigma_vm_expr)

            
            # ----------------------------------------------------------------------------------

            
           
            U = 0.5 * ufl.inner(sigma(uh2_in), strain(uh2_in)) * ufl.dx
    
            # Ensamblar la energía de deformación ----<< 09/02/24 ---<<< OKß
            energy = fem.assemble_scalar(fem.form(U)) 
           # print("Energía de deformación total:", energy)
           # print("Sigma_bar:", sigma_bar_i)
            
            
            
            
            # ----------------------------------------------------------------------------------------------------------
            with XDMFFile(mesh.comm, "./20_01_05_24/fase_3/fem_function_array_modulo.xdmf", "w") as file:
                file.write_mesh(mesh)
                file.write_function(fem_function_modulo_E_f , tmp_cycle)    
            
            with XDMFFile(mesh.comm, "./20_01_05_24/fase_3/fem_function_array_damage.xdmf", "w") as file:
                file.write_mesh(mesh)
                file.write_function(fem_function_damage , tmp_cycle)    
            with XDMFFile(mesh.comm, "./20_01_05_24/fase_2/fem_function_alpha.xdmf", "w") as file:
                  file.write_mesh(mesh)
                  file.write_function(fem_function_alpha , tmp_cycle)    
            with XDMFFile(mesh.comm, "./20_01_05_24/fase_3/fem_function_array_rdot.xdmf", "w") as file:
                file.write_mesh(mesh)
                file.write_function(fem_function_array_rdot , tmp_cycle)    
            with XDMFFile(mesh.comm, "./20_01_05_24/fase_3/fem_function_array_psi.xdmf", "w") as file:
                file.write_mesh(mesh)
                file.write_function(fem_function_array_psi , tmp_cycle)    
            with XDMFFile(mesh.comm, "./20_01_05_24/fase_3/fem_function_array_error.xdmf", "w") as file:
                file.write_mesh(mesh)
                file.write_function(fem_function_array_error , tmp_cycle)    
            
    
           
            #-------------------------------------------------------------------------------------------------
            print("#-------------------------------------------------------------------------------------------------")
            print(f"El valor promedio de daño:{promedios_damage[escalar_contador_total_dias]}")
            print(f"El valor promedio de la densidad:{promedios_rho_cortical_gr_cm_3[escalar_contador_total_dias]}")
            print(f"El valor promedio de la fracción de volumen:{promedios_BVTV[escalar_contador_total_dias]}")
            print(f"El valor promedio de alpha {promedios_alpha[escalar_contador_total_dias]}")
            print(f"3.{dday} - El valor promedio del modulo de elasticidad:{promedios_modulo_elasticidad[escalar_contador_total_dias]}")
            print("#-------------------------------------------------------------------------------------------------")

            
            
        
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_3/fem_function_densidad_fase_3.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_densidad_gr_cm3 , tmp_day)
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_3/fem_function_BVTV_fase_3.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_BVTV , tmp_day)
    
    
    print("---------------------------------------------------------")              
    print("El valor promedio de la BVTV es:", promedios_BVTV[escalar_contador_total_dias])
    print(f" Inicia la cuarta etapa -->> La energía: {energy}")
    print("---------------------------------------------------------")              
    tmp_lenght = escalar_contador_total_dias
    print(f"03-longitud del CSV-->>>>:{tmp_lenght}")


#---------------------------------------------------------------------------------------------------
# Para fase #4  Esta fase dura 100 dias que es tiempo de vida util de la BMU osea durante 100 - 96 durante 4 dias hay 
# remoción y adición de material mismo tiempo.   
#
# Inicia fase 4 Formación + reabsorción
# 
#----------------------------------------------------------------------------------------------
    for dday in range(1, 5):  # Ciclo externo con dtime va de 1 a 5
        
        escalar_contador_total_dias += 1
        escalar_contador_total_dias_modelo += 1
        t_prima_100 = 4 - dday
        t_alpha = 21 + dday
        t_alpha_0= 0,450951684
        tmp_day = dday
        for cycle in range(1000, 10001, 1000):  # Ciclo interno va de 1000 en 1000 hasta 10000
            tmp_cycle = cycle / 1000
    
            expression_array_ei = ((2 * energy) / array_modulo_E_f) ** (1 / 2)
            array_psi = (expression_array_ei * cycle) ** (1 / 4)
            fem_function_array_psi.x.array[:]=array_psi
            
        
            
            # ---------------------------------------------------------------------------------------------------------
            # Cálculo del error.
            # ---------------------------------------------------------------------------------------------------------
            array_error= ((1/array_BVTV)**2)*array_psi - 50
            fem_function_array_error.x.array[:]= array_error

            # ---------------------------------------------------------------------------------------------------------
            # Cálculo del balance focal oseo
            # con el valor de balance focal puedo calcular el valor remodelacion trabecular.
            # ---------------------------------------------------------------------------------------------------------
            array_rdot = np.where(array_error > width,
                      ratecon4 * array_error + (ratecon3 - ratecon4) * width,
                      np.where((array_error <= width) & (array_error >= 0),
                               ratecon3 * array_error,
                               np.where((array_error < 0) & (array_error >= -width),
                                        ratecon2 * array_error,
                                        ratecon1 * array_error + (ratecon1 - ratecon2) * width)))
            
            fem_function_array_rdot.x.array[:]=array_rdot
            #---------------------------------------------------------------------------------------------------------
            
    
            array_porosidad = 1 - array_BVTV
            array_porosity2 = array_porosidad * array_porosidad
            array_porosity3 = array_porosity2 * array_porosidad
            array_porosity4 = array_porosity3 * array_porosidad
            array_porosity5 = array_porosity4 * array_porosidad
            array_Sv = 0.03226 * array_porosidad - 0.09394 * array_porosity2 + 0.13396 * array_porosity3 - 0.10104 * array_porosity4 + 0.02876 * array_porosity5
    
           
            array_derivative_NBMU = array_Sv * array_rdot *escalar_V_bmu * escalar_contador_total_dias_modelo
            #-------------------------------------------------------------------------------------------
            # El proceso de mineralización se lleva a cabo sobre la capa cortical.
            #-------------------------------------------------------------------------------------------
            
            expression_array_dot_vr = array_BVTV * array_derivative_NBMU * escalar_tmp_ar2 /escalar_TR
            expression_array_dot_vb = array_derivative_NBMU * escalar_tmp_af4 /escalar_TF


            
            fem_function_vb_dot.x.array[:] = expression_array_dot_vb
            
            
            escalar_vm = escalar_vm_prim*(t_alpha -12)/escalar_t_prim
            # escalar_alpha = escalar_vm*escalar_rho_mineral /(escalar_rho_mineral * escalar_vm + escalar_rho_org * escalar_v_org)
            # array_alpha.fill(escalar_alpha)
            
            array_COPY_BVTV= array_BVTV
            
            array_formacion_vb[condicion_trabecular]= array_formacion_vb[condicion_trabecular] 
            array_formacion_vb[condicion_cortical] = array_formacion_vb[condicion_cortical] 
    
            
            
            #-------------------------------------------------------------------------------------------
            # Hay que determinar el valor del porcentaje de cenizas.
            #-------------------------------------------------------------------------------------------
           
            array_BVTV[condicion_trabecular] = array_BVTV[condicion_trabecular]
            array_BVTV[condicion_cortical] = array_BVTV[condicion_cortical]  - expression_array_dot_vb[condicion_cortical] + expression_array_dot_vr[condicion_cortical]
            #-------------------------------------------------------------------------------------------------------
            
            array_vm[condicion_trabecular] = (escalar_vm * array_COPY_BVTV[condicion_trabecular] + escalar_vm * array_formacion_vb[condicion_trabecular])/array_BVTV[condicion_trabecular]
            array_vm[condicion_cortical] = (escalar_vm * array_COPY_BVTV[condicion_cortical] - escalar_vm * array_formacion_vb[condicion_cortical] + escalar_vm * expression_array_dot_vr[condicion_cortical])/array_BVTV[condicion_cortical]
           
            promedio_vm_trabe = np.mean(array_vm[condicion_trabecular]) 
            promedio_vm_corti = np.mean(array_vm[condicion_cortical])
            
            print(f"vm_trabecular:{promedio_vm_trabe}")
            print(f"vm_cortical:{promedio_vm_corti}")
            
            array_alpha= array_vm*escalar_rho_mineral/(escalar_rho_mineral*array_vm + escalar_rho_org*escalar_v_org)
            
            
            
            array_BVTV_mean=np.mean(array_BVTV[condicion_cortical])
            array_rho_cortical_gr_cm_3 = array_BVTV[condicion_cortical]*(1.41 +1.29*array_alpha[condicion_cortical]) 
            array_rho_tejido = (1.41 +1.29*array_alpha[condicion_cortical]) 
            promedios_rho_tejido[escalar_contador_total_dias]= np.mean(array_rho_tejido)
            
            array_modulo_E2_cortical = 84370 * (array_BVTV[condicion_cortical] ** 2.58) * (array_alpha[condicion_cortical] ** 2.74) *1e3
            escalar_modulo_E2_cortical = np.mean(array_modulo_E2_cortical)
            #array_modulo_E_f = 84370 * (array_BVTV[condicion_cortical] ** 2.58) * (array_alpha ** 2.74) *1e3

            array_modulo_E_f= 84370 * (array_BVTV ** 2.58) * (array_alpha ** 2.74) *1e3
            array_damage = 1-(array_modulo_E_f/array_modulo_E_0)
            array_damage_cortical = 1 - (escalar_modulo_E2_cortical/escalar_mean_modulo_cortical)
            
            
            promedios_damage[escalar_contador_total_dias]= np.mean(array_damage_cortical)
            promedios_modulo_elasticidad[escalar_contador_total_dias]= escalar_modulo_E2_cortical
            promedios_alpha[escalar_contador_total_dias]=np.mean(array_alpha)
            promedios_rho_cortical_gr_cm_3[escalar_contador_total_dias]= np.mean(array_rho_cortical_gr_cm_3)
            promedios_BVTV[escalar_contador_total_dias]=array_BVTV_mean 
            escalar_energy[escalar_contador_total_dias] = energy


            # ----------------------------------------------------------------------------------------------------------
            # Aca se modifica densidad 
            # ----------------------------------------------------------------------------------------------------------
            array_densidad_material = (1.41 + 1.29 * array_alpha)
            array_densidad = array_densidad_material * array_BVTV
            
            fem_function_densidad.x.array[:] = array_densidad*1e-6
            fem_function_densidad_gr_cm3.x.array[:] = array_densidad
            fem_function_BVTV.x.array[:] = array_BVTV
            fem_function_modulo_E_f.x.array[:] = array_modulo_E_f
            fem_function_damage.x.array[:] = array_damage
            fem_function_alpha.x.array[:] = array_alpha
            fem_function_densidad_material.x.array[:] = array_densidad_material
            
            # # ----------------------------------------------------------------------------------------------------------
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_modulo.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_modulo_E_f , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_densidad_material.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_densidad_material , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_BVTV.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_BVTV , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_array_modulo.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_modulo_E_f , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_array_vm.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_array_vm , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_array_damage.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_damage , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_alpha.xdmf", "w") as file:
            #      file.write_mesh(mesh)
            #      file.write_function(fem_function_alpha , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_array_rdot.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_array_rdot , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_array_psi.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_array_psi , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_array_error.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_array_error , tmp_cycle)    
    
            
            #-------------------------------------------------------------------------------------------------
            ##---------------------------------------------------------------------------------------------------
            ##---------------------------------------------------------------------------------------------------

            # Continúa con la parte de resolución de ecuaciones y condiciones de frontera...
            # Al parecer tambien hay un error en el calculo de mu no es + sino -
            mu = fem_function_modulo_E_f / (2.0 * (1.0 + 0.3))
            lame = fem_function_modulo_E_f * 0.3/ ((1.0 + 0.3) * (1.0 - 2.0 * 0.3)) 
            # revisar el valor de lamnda pues al parecer hay un error  
                                    
            f = ufl.as_vector((0, 0, -fem_function_densidad * 9810 ))
            V = functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
            u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
           
            def sigma(v):
                """Return an expression for the stress σ given a displacement field"""
                return 2.0 * mu * ufl.sym(grad(v)) + lame * ufl.tr(ufl.sym(grad(v))) * ufl.Identity(3)

            def strain(u):
                return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
            a = form(inner(sigma(u), grad(v)) * dx)
            L = form(inner(f, v) * dx)

            def invariants_principal(A):
                i1 = ufl.tr(A)
                i2 = (ufl.tr(A)**2 - ufl.tr(A * A)) / 2
                i3 = ufl.det(A)
                return i1, i2, i3

            #----------------------------------------------------------------------------
            # Asignacion de la condición frontera 08/02/24 - R-ninety
            #----------------------------------------------------------------------------
            def boundary_z(x):
                return np.logical_and(x[2] >= 36, x[2] <= 37)
            
            # Ubicar los grados de libertad en las caras con z entre 36 y 37
            dofs = dolfinx.fem.locate_dofs_geometrical(V, boundary_z)
            
            # Definir la condición de frontera como desplazamiento 0 en todas las direcciones
            u_D = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0, 0, 0)))
            
            # Aplicar la condición de frontera
            bc = dolfinx.fem.dirichletbc(u_D, dofs, V)
            #----------------------------------------------------------------------------
            
            # ## Assemble and solve
            #
            # The bilinear form `a` is assembled into a matrix `A`, with
            # modifications for the Dirichlet boundary conditions. The call
            # `A.assemble()` completes any parallel communication required to
            # compute the matrix.


            A = assemble_matrix(a, bcs=[bc])
            A.assemble()
            # Coordenadas de los puntos donde se aplicarán las cargas puntuales
            point_138 = np.array([87.7431, 20.6717, 224.6261],dtype=np.float64)
            point_140 = np.array([85.6372, 33.9296, 226.9851],dtype=np.float64)
            point_141 = np.array([77.7509, 23.4239, 216.3084],dtype=np.float64)
            point_143 = np.array([78.43, 17.2494, 209.5734],dtype=np.float64)
            point_148 = np.array([73.5285, 30.5294, 203.1597],dtype=np.float64)
            point_52 = np.array([124.2809, 54.4087,180.5598],dtype=np.float64)

            # Cargas en la dirección Z y X

            def marker_52(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_52[0], atol=atol)
                close_y = np.isclose(x[1], point_52[1], atol=atol)
                close_z = np.isclose(x[2], point_52[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_138(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_138[0], atol=atol)
                close_y = np.isclose(x[1], point_138[1], atol=atol)
                close_z = np.isclose(x[2], point_138[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_140(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_140[0], atol=atol)
                close_y = np.isclose(x[1], point_140[1], atol=atol)
                close_z = np.isclose(x[2], point_140[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)
            def marker_141(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_141[0], atol=atol)
                close_y = np.isclose(x[1], point_141[1], atol=atol)
                close_z = np.isclose(x[2], point_141[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)
            def marker_143(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_143[0], atol=atol)
                close_y = np.isclose(x[1], point_143[1], atol=atol)
                close_z = np.isclose(x[2], point_143[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_148(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_148[0], atol=atol)
                close_y = np.isclose(x[1], point_148[1], atol=atol)
                close_z = np.isclose(x[2], point_148[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)



            b = assemble_vector(L)

            # Localiza los grados de libertad en el vértice
            vertices_52 = locate_dofs_geometrical(V, marker_138)
            dofs_52_x = (vertices_52[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_52_z = (vertices_52[0])*3 + 2  # Grado de libertad en la dirección z


            vertices_138 = locate_dofs_geometrical(V, marker_138)
            dofs_138_x = (vertices_138[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_138_z = (vertices_138[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_140 = locate_dofs_geometrical(V, marker_140)
            dofs_140_x = (vertices_140[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_140_z = (vertices_140[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_141 = locate_dofs_geometrical(V, marker_141)
            dofs_141_x = (vertices_141[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_141_z = (vertices_141[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_143 = locate_dofs_geometrical(V, marker_143)
            dofs_143_x = (vertices_143[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_143_z = (vertices_143[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_148 = locate_dofs_geometrical(V, marker_148)
            dofs_148_x = (vertices_148[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_148_z = (vertices_148[0])*3 + 2  # Grado de libertad en la dirección z



            #print("El valor de los DOFs en X y en Z en la  carga:", dofs_138_x, dofs_138_z)
            # Magnitudes de las cargas puntuales
            P_x = 0  # Carga en la dirección y 
            P_z = -2317   # Carga en la dirección z
            P_z2= 713       # Carga del abductor en z
                # Aplicar las cargas puntuales
            with b.localForm() as loc_b:
                loc_b.setValues(dofs_138_x, P_x)
                loc_b.setValues(dofs_138_z, P_z)
                loc_b.setValues(dofs_140_x, P_x)
                loc_b.setValues(dofs_140_z, P_z) 
                loc_b.setValues(dofs_141_x, P_x)
                loc_b.setValues(dofs_141_z, P_z) 
                loc_b.setValues(dofs_143_x, P_x)
                loc_b.setValues(dofs_143_z, P_z) 
                loc_b.setValues(dofs_148_x, P_x)
                loc_b.setValues(dofs_148_z, P_z)
                loc_b.setValues(dofs_52_x, P_x)
                loc_b.setValues(dofs_52_z, P_z2)
            

           
            
            apply_lifting(b, [a], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
            set_bc(b, [bc])
    
    
            ns = build_nullspace(V)
            A.setNearNullSpace(ns)
            A.setOption(PETSc.Mat.Option.SPD, True)  # type: ignore
    
            
            # +
            # Set solver options
            opts = PETSc.Options()  # type: ignore
            opts["ksp_type"] = "cg"
            opts["ksp_rtol"] = 1.0e-8
            opts["pc_type"] = "gamg"
        
            # Use Chebyshev smoothing for multigrid
            opts["mg_levels_ksp_type"] = "chebyshev"
            opts["mg_levels_pc_type"] = "jacobi"
        
            # Improve estimate of eigenvalues for Chebyshev smoothing
            opts["mg_levels_ksp_chebyshev_esteig_steps"] = 10
        
            # Create PETSc Krylov solver and turn convergence monitoring on
            solver = PETSc.KSP().create(mesh.comm)  # type: ignore
            solver.setFromOptions()
    
            # Set matrix operator
            solver.setOperators(A)
            
            # +
            uh2_in = Function(V)
    
            solver.solve(b, uh2_in.vector)
    
            uh2_in.x.scatter_forward()
            
            # ## Post-processing
            # +
            # sigma_dev = sigma(uh_in) - (1 / 3) * ufl.tr(sigma(uh_in)) * ufl.Identity(len(uh_in))
            # sigma_vm = ufl.sqrt((3 / 2) * inner(sigma_dev, sigma_dev))
            # W = functionspace(mesh, ("Discontinuous Lagrange", 0))
            
            # sigma_vm_expr = Expression(sigma_vm, W.element.interpolation_points())
            # sigma_vm_h = Function(W,name="Von_misses")
            # sigma_vm_h.interpolate(sigma_vm_expr)

            
            # ----------------------------------------------------------------------------------

            
           
            U = 0.5 * ufl.inner(sigma(uh2_in), strain(uh2_in)) * ufl.dx
    
            # Ensamblar la energía de deformación ----<< 09/02/24 ---<<< OKß
            energy = fem.assemble_scalar(fem.form(U)) 
           # print("Energía de deformación total:", energy)
           # print("Sigma_bar:", sigma_bar_i)
            print("#-------------------------------------------------------------------------------------------------")
            print(f"4.El valor promedio de daño:{promedios_damage[escalar_contador_total_dias]}")
            print(f"El valor promedio de la densidad:{promedios_rho_cortical_gr_cm_3[escalar_contador_total_dias]}")
            print(f"El valor promedio de la fracción de volumen:{promedios_BVTV[escalar_contador_total_dias]}")
            print(f"4.{dday} - El valor promedio del modulo de elasticidad:{promedios_modulo_elasticidad[escalar_contador_total_dias]}")
            print("#-------------------------------------------------------------------------------------------------")

    # ----------------------------------------------------------------------------------------------------------
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_modulo.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_modulo_E_f , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_densidad_material.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_densidad_material , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_BVTV.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_BVTV , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_array_modulo.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_modulo_E_f , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_array_vm.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_vm , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_array_damage.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_damage , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_alpha.xdmf", "w") as file:
         file.write_mesh(mesh)
         file.write_function(fem_function_alpha , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_array_rdot.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_rdot , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_array_psi.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_psi , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_array_error.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_error , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_densidad_fase_4.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_densidad_gr_cm3 , tmp_day)
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_4/fem_function_BVTV_fase_4.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_BVTV , tmp_day)
    
    
    
    print("---------------------------------------------------------")              
    print("El valor promedio de la BVTV es:", promedios_BVTV[escalar_contador_total_dias])
    print(f" Inicia la quinta etapa - 6 dias restantes -->> La energía: {energy}")
    print("---------------------------------------------------------")              
    tmp_lenght = escalar_contador_total_dias
    print(f"04-longitud del CSV-->>>>:{tmp_lenght}")
    escalar_contador_total_dias_modelo = 0
#-------------------------------------------------------------------------------------------
    for dday in range(1, 7):  # Ciclo externo con dtime va de 1 a 24
        escalar_contador_total_dias += 1
        escalar_contador_total_dias_modelo += 1
        t_prima_100_TR = 6 - dday
        t_alpha = 25 + dday
        tmp_day = dday
        
        for cycle in range(1000, 10001, 1000):  # Ciclo interno va de 1000 en 1000 hasta 10000
            tmp_cycle = cycle / 1000
    
            expression_array_ei = ((2 * energy) / array_modulo_E_f) ** (1 / 2)
            array_psi = (expression_array_ei * cycle) ** (1 / 4)
            fem_function_array_psi.x.array[:]= array_psi
            # ---------------------------------------------------------------------------------------------------------
            # Cálculo del error.
            # ---------------------------------------------------------------------------------------------------------
            array_error= ((1/array_BVTV)**2)*array_psi - 50
            fem_function_array_error.x.array[:]= array_error

            # ---------------------------------------------------------------------------------------------------------
            # Cálculo del balance focal oseo
            # con el valor de balance focal puedo calcular el valor remodelacion trabecular.
            # ---------------------------------------------------------------------------------------------------------
            array_rdot = np.where(array_error > width,
                      ratecon4 * array_error + (ratecon3 - ratecon4) * width,
                      np.where((array_error <= width) & (array_error >= 0),
                               ratecon3 * array_error,
                               np.where((array_error < 0) & (array_error >= -width),
                                        ratecon2 * array_error,
                                        ratecon1 * array_error + (ratecon1 - ratecon2) * width)))
            fem_function_array_rdot.x.array[:]=array_rdot
            
        
            array_porosidad = 1 - array_BVTV
            array_porosity2 = array_porosidad * array_porosidad
            array_porosity3 = array_porosity2 * array_porosidad
            array_porosity4 = array_porosity3 * array_porosidad
            array_porosity5 = array_porosity4 * array_porosidad
            array_Sv = 0.03226 * array_porosidad - 0.09394 * array_porosity2 + 0.13396 * array_porosity3 - 0.10104 * array_porosity4 + 0.02876 * array_porosity5
    
            array_derivative_NBMU = array_Sv * array_rdot * escalar_V_bmu * escalar_contador_total_dias_modelo
    
            escalar_tmp_ar5 = escalar_tmp_ar2 * (1- (t_prima_100_TR / escalar_TR) ** 2)
            expression_array_dot_vr = (array_BVTV) * array_derivative_NBMU * escalar_tmp_ar5 /escalar_TR
#            print(f" dentro de la quinta etapa -->> La expression_array_dot_vr: { escalar_tmp_ar5}")
            array_COPY_BVTV= array_BVTV

            array_formacion_vb[condicion_trabecular]= array_formacion_vb[condicion_trabecular] - (array_derivative_NBMU[condicion_trabecular]/escalar_TR)
            # en la etapa 5 no hay formacion hay reabsortion
            array_formacion_vb[condicion_cortical] = array_formacion_vb[condicion_cortical] + expression_array_dot_vr[condicion_cortical] 
      
              
    
            # Asegúrate de definir `expression_array_dot_vb` correctamente si se usa aquí
            array_BVTV[condicion_trabecular] = array_BVTV[condicion_trabecular] - array_derivative_NBMU[condicion_trabecular]/escalar_TR
            array_BVTV[condicion_cortical]= array_BVTV[condicion_cortical] + expression_array_dot_vr[condicion_cortical]
            #-------------------------------------------------------------------------------------------------------
            escalar_vm = escalar_vm_prim*(t_alpha -12)/escalar_t_prim

            array_vm[condicion_trabecular] = (escalar_vm * array_COPY_BVTV[condicion_trabecular] + escalar_vm * array_formacion_vb[condicion_trabecular])/array_BVTV[condicion_trabecular]
            array_vm[condicion_cortical] = (escalar_vm * array_COPY_BVTV[condicion_cortical] + escalar_vm*array_formacion_vb[condicion_cortical])/array_BVTV[condicion_cortical]
           
            promedio_vm_trabe = np.mean(array_vm[condicion_trabecular]) 
            promedio_vm_corti = np.mean(array_vm[condicion_cortical])
            
            print(f"vm_trabecular:{promedio_vm_trabe}")
            print(f"vm_cortical:{promedio_vm_corti}")
            
            array_alpha= array_vm*escalar_rho_mineral/(escalar_rho_mineral*array_vm + escalar_rho_org*escalar_v_org)
            
            
            
            array_BVTV_mean=np.mean(array_BVTV[condicion_cortical])
            
            
        
            array_rho_cortical_gr_cm_3 = array_BVTV[condicion_cortical]*(1.41 +1.29*array_alpha[condicion_cortical]) 
            array_rho_tejido = (1.41 +1.29*array_alpha[condicion_cortical]) 
            promedios_rho_tejido[escalar_contador_total_dias]= np.mean(array_rho_tejido)
            
            array_modulo_E2_cortical = 84370 * (array_BVTV[condicion_cortical] ** 2.58) * (array_alpha[condicion_cortical] ** 2.74) *1e3
            escalar_modulo_E2_cortical = np.mean(array_modulo_E2_cortical)
#            array_modulo_E_f[condicion_cortical]= 84370 * (array_BVTV[condicion_cortical] ** 2.58) * (array_alpha ** 2.74) *1e3
            array_modulo_E_f = 84370 * (array_BVTV ** 2.58) * (array_alpha ** 2.74) *1e3

            array_damage = 1-(array_modulo_E_f/array_modulo_E_0)
            array_damage_cortical = 1 - (escalar_modulo_E2_cortical/escalar_mean_modulo_cortical)
            
            
            promedios_damage[escalar_contador_total_dias]= np.mean(array_damage_cortical)
            promedios_modulo_elasticidad[escalar_contador_total_dias]= escalar_modulo_E2_cortical
            promedios_rho_cortical_gr_cm_3[escalar_contador_total_dias]= np.mean(array_rho_cortical_gr_cm_3)
            promedios_BVTV[escalar_contador_total_dias]=array_BVTV_mean 
            escalar_energy[escalar_contador_total_dias] = energy
            promedios_alpha[escalar_contador_total_dias]=np.mean(array_alpha)

            


            # ----------------------------------------------------------------------------------------------------------
            # Aca se modifica densidad 
            # ----------------------------------------------------------------------------------------------------------
            array_densidad_material = (1.41 + 1.29 * array_alpha)
            array_densidad = array_densidad_material * array_BVTV

            
            fem_function_densidad.x.array[:] = array_densidad*1e-6
            fem_function_densidad_gr_cm3.x.array[:] = array_densidad
            fem_function_BVTV.x.array[:] = array_BVTV
            fem_function_modulo_E_f.x.array[:] = array_modulo_E_f
            fem_function_damage.x.array[:] = array_damage
            fem_function_alpha.x.array[:] = array_alpha
            fem_function_densidad_material.x.array[:] = array_densidad_material
            
            # # ----------------------------------------------------------------------------------------------------------
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_modulo.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_modulo_E_f , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_densidad_material.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_densidad_material , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_array_damage.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_damage , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_array_alpha.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_alpha , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_array_rdot.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_array_rdot , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_array_psi.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_array_psi , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_array_error.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_array_error , tmp_cycle)    
    
            
            #-------------------------------------------------------------------------------------------------
            ##---------------------------------------------------------------------------------------------------
            ##---------------------------------------------------------------------------------------------------

            # Continúa con la parte de resolución de ecuaciones y condiciones de frontera...
            # Al parecer tambien hay un error en el calculo de mu no es + sino -
            mu = fem_function_modulo_E_f / (2.0 * (1.0 + 0.3))
            lame = fem_function_modulo_E_f * 0.3/ ((1.0 + 0.3) * (1.0 - 2.0 * 0.3)) 
            # revisar el valor de lamnda pues al parecer hay un error  
                                    
            f = ufl.as_vector((0, 0, -fem_function_densidad * 9810 ))
            V = functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
            u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
           
            def sigma(v):
                """Return an expression for the stress σ given a displacement field"""
                return 2.0 * mu * ufl.sym(grad(v)) + lame * ufl.tr(ufl.sym(grad(v))) * ufl.Identity(3)

            def strain(u):
                return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
            a = form(inner(sigma(u), grad(v)) * dx)
            L = form(inner(f, v) * dx)

            def invariants_principal(A):
                i1 = ufl.tr(A)
                i2 = (ufl.tr(A)**2 - ufl.tr(A * A)) / 2
                i3 = ufl.det(A)
                return i1, i2, i3

            #----------------------------------------------------------------------------
            # Asignacion de la condición frontera 08/02/24 - R-ninety
            #----------------------------------------------------------------------------
            def boundary_z(x):
                return np.logical_and(x[2] >= 36, x[2] <= 37)
            
            # Ubicar los grados de libertad en las caras con z entre 36 y 37
            dofs = dolfinx.fem.locate_dofs_geometrical(V, boundary_z)
            
            # Definir la condición de frontera como desplazamiento 0 en todas las direcciones
            u_D = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0, 0, 0)))
            
            # Aplicar la condición de frontera
            bc = dolfinx.fem.dirichletbc(u_D, dofs, V)
            #----------------------------------------------------------------------------
            
            # ## Assemble and solve
            #
            # The bilinear form `a` is assembled into a matrix `A`, with
            # modifications for the Dirichlet boundary conditions. The call
            # `A.assemble()` completes any parallel communication required to
            # compute the matrix.


            A = assemble_matrix(a, bcs=[bc])
            A.assemble()
            
            

            # Coordenadas de los puntos donde se aplicarán las cargas puntuales
            point_138 = np.array([87.7431, 20.6717, 224.6261],dtype=np.float64)
            point_140 = np.array([85.6372, 33.9296, 226.9851],dtype=np.float64)
            point_141 = np.array([77.7509, 23.4239, 216.3084],dtype=np.float64)
            point_143 = np.array([78.43, 17.2494, 209.5734],dtype=np.float64)
            point_148 = np.array([73.5285, 30.5294, 203.1597],dtype=np.float64)
            point_52 = np.array([124.2809, 54.4087,180.5598],dtype=np.float64)

            # Cargas en la dirección Z y X

            def marker_52(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_52[0], atol=atol)
                close_y = np.isclose(x[1], point_52[1], atol=atol)
                close_z = np.isclose(x[2], point_52[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_138(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_138[0], atol=atol)
                close_y = np.isclose(x[1], point_138[1], atol=atol)
                close_z = np.isclose(x[2], point_138[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_140(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_140[0], atol=atol)
                close_y = np.isclose(x[1], point_140[1], atol=atol)
                close_z = np.isclose(x[2], point_140[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)
            def marker_141(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_141[0], atol=atol)
                close_y = np.isclose(x[1], point_141[1], atol=atol)
                close_z = np.isclose(x[2], point_141[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)
            def marker_143(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_143[0], atol=atol)
                close_y = np.isclose(x[1], point_143[1], atol=atol)
                close_z = np.isclose(x[2], point_143[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_148(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_148[0], atol=atol)
                close_y = np.isclose(x[1], point_148[1], atol=atol)
                close_z = np.isclose(x[2], point_148[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)



            b = assemble_vector(L)

            # Localiza los grados de libertad en el vértice
            vertices_52 = locate_dofs_geometrical(V, marker_138)
            dofs_52_x = (vertices_52[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_52_z = (vertices_52[0])*3 + 2  # Grado de libertad en la dirección z


            vertices_138 = locate_dofs_geometrical(V, marker_138)
            dofs_138_x = (vertices_138[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_138_z = (vertices_138[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_140 = locate_dofs_geometrical(V, marker_140)
            dofs_140_x = (vertices_140[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_140_z = (vertices_140[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_141 = locate_dofs_geometrical(V, marker_141)
            dofs_141_x = (vertices_141[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_141_z = (vertices_141[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_143 = locate_dofs_geometrical(V, marker_143)
            dofs_143_x = (vertices_143[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_143_z = (vertices_143[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_148 = locate_dofs_geometrical(V, marker_148)
            dofs_148_x = (vertices_148[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_148_z = (vertices_148[0])*3 + 2  # Grado de libertad en la dirección z



            print("El valor de los DOFs en X y en Z en la  carga:", dofs_138_x, dofs_138_z)
            # Magnitudes de las cargas puntuales
            P_x = 0  # Carga en la dirección y 
            P_z = -2317   # Carga en la dirección z
            P_z2= 713       # Carga del abductor en z
                # Aplicar las cargas puntuales
            with b.localForm() as loc_b:
                loc_b.setValues(dofs_138_x, P_x)
                loc_b.setValues(dofs_138_z, P_z)
                loc_b.setValues(dofs_140_x, P_x)
                loc_b.setValues(dofs_140_z, P_z) 
                loc_b.setValues(dofs_141_x, P_x)
                loc_b.setValues(dofs_141_z, P_z) 
                loc_b.setValues(dofs_143_x, P_x)
                loc_b.setValues(dofs_143_z, P_z) 
                loc_b.setValues(dofs_148_x, P_x)
                loc_b.setValues(dofs_148_z, P_z)
                loc_b.setValues(dofs_52_x, P_x)
                loc_b.setValues(dofs_52_z, P_z2)
            
            
            apply_lifting(b, [a], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
            set_bc(b, [bc])
    
    
            ns = build_nullspace(V)
            A.setNearNullSpace(ns)
            A.setOption(PETSc.Mat.Option.SPD, True)  # type: ignore
    
            
            # +
            # Set solver options
            opts = PETSc.Options()  # type: ignore
            opts["ksp_type"] = "cg"
            opts["ksp_rtol"] = 1.0e-8
            opts["pc_type"] = "gamg"
        
            # Use Chebyshev smoothing for multigrid
            opts["mg_levels_ksp_type"] = "chebyshev"
            opts["mg_levels_pc_type"] = "jacobi"
        
            # Improve estimate of eigenvalues for Chebyshev smoothing
            opts["mg_levels_ksp_chebyshev_esteig_steps"] = 10
        
            # Create PETSc Krylov solver and turn convergence monitoring on
            solver = PETSc.KSP().create(mesh.comm)  # type: ignore
            solver.setFromOptions()
    
            # Set matrix operator
            solver.setOperators(A)
            
            # +
            uh2_in = Function(V)
    
            solver.solve(b, uh2_in.vector)
    
            uh2_in.x.scatter_forward()
            
            # ## Post-processing
            # +
            # sigma_dev = sigma(uh_in) - (1 / 3) * ufl.tr(sigma(uh_in)) * ufl.Identity(len(uh_in))
            # sigma_vm = ufl.sqrt((3 / 2) * inner(sigma_dev, sigma_dev))
            # W = functionspace(mesh, ("Discontinuous Lagrange", 0))
            
            # sigma_vm_expr = Expression(sigma_vm, W.element.interpolation_points())
            # sigma_vm_h = Function(W,name="Von_misses")
            # sigma_vm_h.interpolate(sigma_vm_expr)

            
            # ----------------------------------------------------------------------------------

            
           
            U = 0.5 * ufl.inner(sigma(uh2_in), strain(uh2_in)) * ufl.dx
    
            # Ensamblar la energía de deformación ----<< 09/02/24 ---<<< OKß
            energy = fem.assemble_scalar(fem.form(U)) 
           # print("Energía de deformación total:", energy)
           # print("Sigma_bar:", sigma_bar_i)
            
            
            
            
            
            print("#-------------------------------------------------------------------------------------------------")
            print(f"5.1 {dday} - El valor promedio de daño:{promedios_damage[escalar_contador_total_dias]}")
            print(f"5.1 {dday} - El valor promedio de la densidad:{promedios_rho_cortical_gr_cm_3[escalar_contador_total_dias]}")
            print(f"5.1 {dday} - El valor promedio de la fracción de volumen:{promedios_BVTV[escalar_contador_total_dias]}")
            print(f"5.1 {dday} - El valor promedio del modulo de elasticidad:{promedios_modulo_elasticidad[escalar_contador_total_dias]}")
            print("#-------------------------------------------------------------------------------------------------")

    with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_modulo.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_modulo_E_f , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_densidad_material.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_densidad_material , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_BVTV.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_BVTV , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_array_modulo.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_modulo_E_f , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_array_vm.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_vm , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_array_damage.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_damage , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_alpha.xdmf", "w") as file:
         file.write_mesh(mesh)
         file.write_function(fem_function_alpha , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_array_rdot.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_rdot , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_array_psi.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_psi , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_array_error.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_error , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_densidad_fase_4.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_densidad_gr_cm3 , tmp_day)
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_BVTV_fase_4.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_BVTV , tmp_day)
        
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_densidad_fase_5.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_densidad_gr_cm3 , tmp_day)
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_BVTV_fase_5.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_BVTV , tmp_day)
    print("---------------------------------------------------------")              
    print("El valor promedio de la BVTV es:", promedios_BVTV[escalar_contador_total_dias])
    print(f"Terminan 6  la quinta etapa -->> La energía: {energy}")
    print("---------------------------------------------------------")              
    tmp_lenght = escalar_contador_total_dias
    print(f"05-longitud del CSV-->>>>:{tmp_lenght}")
# -----------------------------------------------------------------------------------
# Aca empiezan los 18 dias restantes de la etapa 5 en donde no hay mineralización    
# ------------------------------------------------------------------------------------
    print("---------------------------------------------------------")              
    print("El valor promedio de la BVTV es:", promedios_BVTV[escalar_contador_total_dias])
    print(f" empiezan la quinta etapa  18 dias --->> La energía: {energy}")
    print("---------------------------------------------------------")              
    tmp_lenght = escalar_contador_total_dias
    print(f"5 -18 dias restantes-longitud del CSV-->>>>:{tmp_lenght}")
    
#-------------------------------------------------------------------------------------------
    for dday in range(1, 19):  # Ciclo externo con dtime va de 1 a 24
        escalar_contador_total_dias += 1
        escalar_contador_total_dias_modelo += 1
        t_prima_100_TR = 18 - dday
        tmp_day = dday
        t_alpha = 31 + dday

        
        for cycle in range(1000, 10001, 1000):  # Ciclo interno va de 1000 en 1000 hasta 10000
            tmp_cycle = cycle / 1000
    
            expression_array_ei = ((2 * energy) / array_modulo_E_f) ** (1 / 2)
            array_psi = (expression_array_ei * cycle) ** (1 / 4)
            fem_function_array_psi.x.array[:]= array_psi
            # ---------------------------------------------------------------------------------------------------------
            # Cálculo del error.
            # ---------------------------------------------------------------------------------------------------------
            array_error= ((1/array_BVTV)**2)*array_psi - 50
            fem_function_array_error.x.array[:]= array_error

            # ---------------------------------------------------------------------------------------------------------
            # Cálculo del balance focal oseo
            # con el valor de balance focal puedo calcular el valor remodelacion trabecular.
            # ---------------------------------------------------------------------------------------------------------
            array_rdot = np.where(array_error > width,
                      ratecon4 * array_error + (ratecon3 - ratecon4) * width,
                      np.where((array_error <= width) & (array_error >= 0),
                               ratecon3 * array_error,
                               np.where((array_error < 0) & (array_error >= -width),
                                        ratecon2 * array_error,
                                        ratecon1 * array_error + (ratecon1 - ratecon2) * width)))
            fem_function_array_rdot.x.array[:]=array_rdot
            
        
            array_porosidad = 1 - array_BVTV
            array_porosity2 = array_porosidad * array_porosidad
            array_porosity3 = array_porosity2 * array_porosidad
            array_porosity4 = array_porosity3 * array_porosidad
            array_porosity5 = array_porosity4 * array_porosidad
            array_Sv = 0.03226 * array_porosidad - 0.09394 * array_porosity2 + 0.13396 * array_porosity3 - 0.10104 * array_porosity4 + 0.02876 * array_porosity5
    
            array_derivative_NBMU = array_Sv * array_rdot *escalar_V_bmu * escalar_contador_total_dias_modelo
    
            escalar_tmp_ar5 = escalar_tmp_ar2 * (1 -(t_prima_100_TR / escalar_TR) ** 2)/ escalar_TR
            expression_array_dot_vr = (array_BVTV) * array_derivative_NBMU * escalar_tmp_ar5 
            
            escalar_vm = escalar_vm_max - (escalar_vm_max -escalar_vm_prim)*math.exp(-escalar_km*(t_alpha -escalar_t_prim -escalar_t_nm))
            
            array_formacion_vb[condicion_trabecular]= array_formacion_vb[condicion_trabecular] - (array_derivative_NBMU[condicion_trabecular]/escalar_TR)
            # en la etapa 5 no hay formacion hay reabsortion
            array_formacion_vb[condicion_cortical] = array_formacion_vb[condicion_cortical] + expression_array_dot_vr[condicion_cortical] 
      
            
            array_COPY_BVTV= array_BVTV
           
            promedio_vm_trabe = np.mean(array_vm[condicion_trabecular]) 
            promedio_vm_corti = np.mean(array_vm[condicion_cortical])
            
            print(f"vm_trabecular:{promedio_vm_trabe}")
            print(f"vm_cortical:{promedio_vm_corti}")
            
            
    
            # Asegúrate de definir `expression_array_dot_vb` correctamente si se usa aquí
            array_BVTV[condicion_trabecular] = array_BVTV[condicion_trabecular] - array_derivative_NBMU[condicion_trabecular]/escalar_TR
            array_BVTV[condicion_cortical]= array_BVTV[condicion_cortical] + expression_array_dot_vr[condicion_cortical]
            
            array_vm[condicion_trabecular] = (escalar_vm * array_COPY_BVTV[condicion_trabecular] - escalar_vm * array_formacion_vb[condicion_trabecular])/array_BVTV[condicion_trabecular]
            array_vm[condicion_cortical] = (escalar_vm * array_COPY_BVTV[condicion_cortical] + escalar_vm*array_formacion_vb[condicion_cortical])/array_BVTV[condicion_cortical]
           
            array_alpha= array_vm * escalar_rho_mineral / (escalar_rho_mineral*array_vm + escalar_rho_org*escalar_v_org)

            
            
            #-------------------------------------------------------------------------------------------------------
            array_BVTV_mean=np.mean(array_BVTV[condicion_cortical])
            
            array_rho_cortical_gr_cm_3 = array_BVTV[condicion_cortical]*(1.41 +1.29*array_alpha[condicion_cortical]) 
            array_rho_tejido = (1.41 +1.29*array_alpha[condicion_cortical]) 
            promedios_rho_tejido[escalar_contador_total_dias]= np.mean(array_rho_tejido)
            
            array_modulo_E2_cortical = 84370 * (array_BVTV[condicion_cortical] ** 2.58) * (array_alpha[condicion_cortical] ** 2.74) *1e3
            escalar_modulo_E2_cortical = np.mean(array_modulo_E2_cortical)
#            array_modulo_E_f[condicion_cortical]= 84370 * (array_BVTV[condicion_cortical] ** 2.58) * (array_alpha ** 2.74) *1e3
            array_modulo_E_f = 84370 * (array_BVTV ** 2.58) * (array_alpha ** 2.74) *1e3

            array_damage = 1-(array_modulo_E_f/array_modulo_E_0)
            array_damage_cortical = 1 - (escalar_modulo_E2_cortical/escalar_mean_modulo_cortical)
            
            promedios_damage[escalar_contador_total_dias]= np.mean(array_damage_cortical)
            promedios_modulo_elasticidad[escalar_contador_total_dias]= escalar_modulo_E2_cortical
            promedios_rho_cortical_gr_cm_3[escalar_contador_total_dias]= np.mean(array_rho_cortical_gr_cm_3)
            promedios_BVTV[escalar_contador_total_dias]=array_BVTV_mean 
            escalar_energy[escalar_contador_total_dias] = energy
            promedios_alpha[escalar_contador_total_dias]=np.mean(array_alpha)

            
            # ----------------------------------------------------------------------------------------------------------
            # Aca se modifica densidad 
            # ----------------------------------------------------------------------------------------------------------
            #array_densidad[condicion_cortical] = (1.41 + 1.29 * alpha_ini) * array_BVTV[condicion_cortical]
            # Aca se modifica densidad 
            array_densidad_material = (1.41 + 1.29 * array_alpha) 
            array_densidad = array_densidad_material * array_BVTV
            
            fem_function_densidad.x.array[:] = array_densidad*1e-6
            fem_function_densidad_gr_cm3.x.array[:] = array_densidad
            fem_function_BVTV.x.array[:] = array_BVTV
            fem_function_modulo_E_f.x.array[:] = array_modulo_E_f
            fem_function_damage.x.array[:] = array_damage
            fem_function_alpha.x.array[:]= array_alpha
            fem_function_densidad_material.x.array[:] = array_densidad_material
            # ----------------------------------------------------------------------------------------------------------
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_array_modulo.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_modulo_E_f , tmp_cycle)    
            
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_array_damage.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_damage , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_array_alpha.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_alpha , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_array_rdot.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_array_rdot , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_array_psi.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_array_psi , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_5/fem_function_array_error.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_array_error , tmp_cycle)    
    
            
            #-------------------------------------------------------------------------------------------------
            ##---------------------------------------------------------------------------------------------------
            ##---------------------------------------------------------------------------------------------------

            # Continúa con la parte de resolución de ecuaciones y condiciones de frontera...
            # Al parecer tambien hay un error en el calculo de mu no es + sino -
            mu = fem_function_modulo_E_f / (2.0 * (1.0 + 0.3))
            lame = fem_function_modulo_E_f * 0.3/ ((1.0 + 0.3) * (1.0 - 2.0 * 0.3)) 
            # revisar el valor de lamnda pues al parecer hay un error  
                                    
            f = ufl.as_vector((0, 0, -fem_function_densidad * 9810 ))
            V = functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
            u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
           
            def sigma(v):
                """Return an expression for the stress σ given a displacement field"""
                return 2.0 * mu * ufl.sym(grad(v)) + lame * ufl.tr(ufl.sym(grad(v))) * ufl.Identity(3)

            def strain(u):
                return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
            a = form(inner(sigma(u), grad(v)) * dx)
            L = form(inner(f, v) * dx)

            def invariants_principal(A):
                i1 = ufl.tr(A)
                i2 = (ufl.tr(A)**2 - ufl.tr(A * A)) / 2
                i3 = ufl.det(A)
                return i1, i2, i3

            #----------------------------------------------------------------------------
            # Asignacion de la condición frontera 08/02/24 - R-ninety
            #----------------------------------------------------------------------------
            def boundary_z(x):
                return np.logical_and(x[2] >= 36, x[2] <= 37)
            
            # Ubicar los grados de libertad en las caras con z entre 36 y 37
            dofs = dolfinx.fem.locate_dofs_geometrical(V, boundary_z)
            
            # Definir la condición de frontera como desplazamiento 0 en todas las direcciones
            u_D = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0, 0, 0)))
            
            # Aplicar la condición de frontera
            bc = dolfinx.fem.dirichletbc(u_D, dofs, V)
            #----------------------------------------------------------------------------
            
            # ## Assemble and solve
            #
            # The bilinear form `a` is assembled into a matrix `A`, with
            # modifications for the Dirichlet boundary conditions. The call
            # `A.assemble()` completes any parallel communication required to
            # compute the matrix.


            A = assemble_matrix(a, bcs=[bc])
            A.assemble()
            
            

            # Coordenadas de los puntos donde se aplicarán las cargas puntuales
            point_138 = np.array([87.7431, 20.6717, 224.6261],dtype=np.float64)
            point_140 = np.array([85.6372, 33.9296, 226.9851],dtype=np.float64)
            point_141 = np.array([77.7509, 23.4239, 216.3084],dtype=np.float64)
            point_143 = np.array([78.43, 17.2494, 209.5734],dtype=np.float64)
            point_148 = np.array([73.5285, 30.5294, 203.1597],dtype=np.float64)
            point_52 = np.array([124.2809, 54.4087,180.5598],dtype=np.float64)

            # Cargas en la dirección Z y X

            def marker_52(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_52[0], atol=atol)
                close_y = np.isclose(x[1], point_52[1], atol=atol)
                close_z = np.isclose(x[2], point_52[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_138(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_138[0], atol=atol)
                close_y = np.isclose(x[1], point_138[1], atol=atol)
                close_z = np.isclose(x[2], point_138[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_140(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_140[0], atol=atol)
                close_y = np.isclose(x[1], point_140[1], atol=atol)
                close_z = np.isclose(x[2], point_140[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)
            def marker_141(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_141[0], atol=atol)
                close_y = np.isclose(x[1], point_141[1], atol=atol)
                close_z = np.isclose(x[2], point_141[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)
            def marker_143(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_143[0], atol=atol)
                close_y = np.isclose(x[1], point_143[1], atol=atol)
                close_z = np.isclose(x[2], point_143[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_148(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_148[0], atol=atol)
                close_y = np.isclose(x[1], point_148[1], atol=atol)
                close_z = np.isclose(x[2], point_148[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)



            b = assemble_vector(L)

            # Localiza los grados de libertad en el vértice
            vertices_52 = locate_dofs_geometrical(V, marker_138)
            dofs_52_x = (vertices_52[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_52_z = (vertices_52[0])*3 + 2  # Grado de libertad en la dirección z


            vertices_138 = locate_dofs_geometrical(V, marker_138)
            dofs_138_x = (vertices_138[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_138_z = (vertices_138[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_140 = locate_dofs_geometrical(V, marker_140)
            dofs_140_x = (vertices_140[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_140_z = (vertices_140[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_141 = locate_dofs_geometrical(V, marker_141)
            dofs_141_x = (vertices_141[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_141_z = (vertices_141[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_143 = locate_dofs_geometrical(V, marker_143)
            dofs_143_x = (vertices_143[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_143_z = (vertices_143[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_148 = locate_dofs_geometrical(V, marker_148)
            dofs_148_x = (vertices_148[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_148_z = (vertices_148[0])*3 + 2  # Grado de libertad en la dirección z



            print("El valor de los DOFs en X y en Z en la  carga:", dofs_138_x, dofs_138_z)
            # Magnitudes de las cargas puntuales
            P_x = 0  # Carga en la dirección y 
            P_z = -2317   # Carga en la dirección z
            P_z2= 713       # Carga del abductor en z
                # Aplicar las cargas puntuales
            with b.localForm() as loc_b:
                loc_b.setValues(dofs_138_x, P_x)
                loc_b.setValues(dofs_138_z, P_z)
                loc_b.setValues(dofs_140_x, P_x)
                loc_b.setValues(dofs_140_z, P_z) 
                loc_b.setValues(dofs_141_x, P_x)
                loc_b.setValues(dofs_141_z, P_z) 
                loc_b.setValues(dofs_143_x, P_x)
                loc_b.setValues(dofs_143_z, P_z) 
                loc_b.setValues(dofs_148_x, P_x)
                loc_b.setValues(dofs_148_z, P_z)
                loc_b.setValues(dofs_52_x, P_x)
                loc_b.setValues(dofs_52_z, P_z2)
            
            
            apply_lifting(b, [a], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
            set_bc(b, [bc])
    
    
            ns = build_nullspace(V)
            A.setNearNullSpace(ns)
            A.setOption(PETSc.Mat.Option.SPD, True)  # type: ignore
    
            
            # +
            # Set solver options
            opts = PETSc.Options()  # type: ignore
            opts["ksp_type"] = "cg"
            opts["ksp_rtol"] = 1.0e-8
            opts["pc_type"] = "gamg"
        
            # Use Chebyshev smoothing for multigrid
            opts["mg_levels_ksp_type"] = "chebyshev"
            opts["mg_levels_pc_type"] = "jacobi"
        
            # Improve estimate of eigenvalues for Chebyshev smoothing
            opts["mg_levels_ksp_chebyshev_esteig_steps"] = 10
        
            # Create PETSc Krylov solver and turn convergence monitoring on
            solver = PETSc.KSP().create(mesh.comm)  # type: ignore
            solver.setFromOptions()
    
            # Set matrix operator
            solver.setOperators(A)
            
            # +
            uh2_in = Function(V)
    
            solver.solve(b, uh2_in.vector)
    
            uh2_in.x.scatter_forward()
            
            # ## Post-processing
            # +
            # sigma_dev = sigma(uh_in) - (1 / 3) * ufl.tr(sigma(uh_in)) * ufl.Identity(len(uh_in))
            # sigma_vm = ufl.sqrt((3 / 2) * inner(sigma_dev, sigma_dev))
            # W = functionspace(mesh, ("Discontinuous Lagrange", 0))
            
            # sigma_vm_expr = Expression(sigma_vm, W.element.interpolation_points())
            # sigma_vm_h = Function(W,name="Von_misses")
            # sigma_vm_h.interpolate(sigma_vm_expr)

            
            # ----------------------------------------------------------------------------------

            
           
            U = 0.5 * ufl.inner(sigma(uh2_in), strain(uh2_in)) * ufl.dx
    
            # Ensamblar la energía de deformación ----<< 09/02/24 ---<<< OKß
            energy = fem.assemble_scalar(fem.form(U)) 
           # print("Energía de deformación total:", energy)
           # print("Sigma_bar:", sigma_bar_i)
            
            
            
            
            
            print("#-------------------------------------------------------------------------------------------------")
            print(f"5.2 {dday} - El valor promedio de daño:{promedios_damage[escalar_contador_total_dias]}")
            print(f"5.2 {dday} - El valor promedio de la densidad:{promedios_rho_cortical_gr_cm_3[escalar_contador_total_dias]}")
            print(f"5.2 {dday} - El valor promedio de la fracción de volumen:{promedios_BVTV[escalar_contador_total_dias]}")
            print(f"5.2 {dday} - El valor promedio del modulo de elasticidad:{promedios_modulo_elasticidad[escalar_contador_total_dias]}")
            print("#-------------------------------------------------------------------------------------------------")

    with XDMFFile(mesh.comm, "./20_01_05_24/fase_52/fem_function_modulo.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_modulo_E_f , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_52/fem_function_densidad_material.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_densidad_material , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_52/fem_function_BVTV.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_BVTV , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_52/fem_function_array_modulo.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_modulo_E_f , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_52/fem_function_array_vm.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_vm , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_52/fem_function_array_damage.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_damage , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_52/fem_function_alpha.xdmf", "w") as file:
         file.write_mesh(mesh)
         file.write_function(fem_function_alpha , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_52/fem_function_array_rdot.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_rdot , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_52/fem_function_array_psi.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_psi , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_52/fem_function_array_error.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_error , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_52/fem_function_densidad_fase_52.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_densidad_gr_cm3 , tmp_day)
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_52/fem_function_BVTV_fase_52.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_BVTV , tmp_day)

    print("---------------------------------------------------------")              
    print("El valor promedio de la BVTV es:", promedios_BVTV[escalar_contador_total_dias])
    print(f"Inicia la sexta etapa -->> La energía: {energy}")
    print("---------------------------------------------------------")              
    tmp_lenght = escalar_contador_total_dias
    print(f"05-longitud del CSV-->>>>:{tmp_lenght}")
#--


    
    
    
    
    
    
#-------------------------------------------------------------------------------------------
    # Para fase #6 Esta fase dura 8 dias + TR + TI
#-------------------------------------------------------------------------------------------
    for dday in range(1, 9):  # Ciclo externo con dtime va de 1 a 32
        t_prima_100_TR_TI = 8 - dday
        escalar_contador_total_dias += 1
        escalar_contador_total_dias_modelo += 1
        tmp_day = dday
        t_alpha = 49 + dday

        # Itera sobre los ciclos de carga
        for cycle in range(1000, 10001, 1000):  # Ciclo interno va de 1000 en 1000 hasta 10000

            tmp_cycle = cycle / 1000
            #print(f"energy {dmes}:{energy}")
    
            # Calcula el índice de daño basado en la energía y el módulo de elasticidad
            expression_array_ei = ((2 * energy) / array_modulo_E_f) ** (1 / 2)
            array_psi = (expression_array_ei * cycle) ** (1 / 4)
            fem_function_array_psi.x.array[:] = array_psi
            # ---------------------------------------------------------------------------------------------------------
            # Cálculo del error.
            # ---------------------------------------------------------------------------------------------------------
            array_error= ((1/array_BVTV)**2)*array_psi - 50
            fem_function_array_error.x.array[:]= array_error

            # ---------------------------------------------------------------------------------------------------------
            # Cálculo del balance focal oseo
            # con el valor de balance focal puedo calcular el valor remodelacion trabecular.
            # ---------------------------------------------------------------------------------------------------------
            array_rdot = np.where(array_error > width,
                      ratecon4 * array_error + (ratecon3 - ratecon4) * width,
                      np.where((array_error <= width) & (array_error >= 0),
                               ratecon3 * array_error,
                               np.where((array_error < 0) & (array_error >= -width),
                                        ratecon2 * array_error,
                                        ratecon1 * array_error + (ratecon1 - ratecon2) * width)))
            fem_function_array_rdot.x.array[:]=array_rdot
            
            
            # Imprime el valor actual de la energía
           # print(f"El valor del escalar de energia [U] en el día {dday} en el ciclo {tmp_cycle} es: {energy}")
            
           
            # Determina la frecuencia de nacimiento de nuevas unidades basales
            
            # Calcula la porosidad basada en la fracción de volumen óseo
            array_porosidad = 1 - array_BVTV
            array_porosity2 = array_porosidad * array_porosidad
            array_porosity3 = array_porosity2 * array_porosidad
            array_porosity4 = array_porosity3 * array_porosidad
            array_porosity5 = array_porosity4 * array_porosidad
            array_Sv = 0.03226 * array_porosidad - 0.09394 * array_porosity2 + 0.13396 * array_porosity3 - 0.10104 * array_porosity4 + 0.02876 * array_porosity5
            
            # Calcula la variación de las unidades basales
            array_derivative_NBMU = array_Sv * array_rdot * escalar_V_bmu * escalar_contador_total_dias_modelo
        
            array_COPY_BVTV = array_BVTV
            
            # Determina el volumen de hueso basado en las unidades basales y la porosidad
            expression_array_dot_vb = array_derivative_NBMU * escalar_tmp_af4 / escalar_TF
            
            array_formacion_vb[condicion_trabecular]= array_formacion_vb[condicion_trabecular] 
            # en la etapa 5 no hay formacion hay reabsortion
            array_formacion_vb[condicion_cortical] = array_formacion_vb[condicion_cortical] - expression_array_dot_vb[condicion_cortical] 
      
            
            escalar_vm = escalar_vm_max - (escalar_vm_max -escalar_vm_prim)*math.exp(-escalar_km*(t_alpha -escalar_t_prim -escalar_t_nm))
           
            promedio_vm_trabe = np.mean(array_vm[condicion_trabecular]) 
            promedio_vm_corti = np.mean(array_vm[condicion_cortical])
            
            print(f"vm_trabecular:{promedio_vm_trabe}")
            print(f"vm_cortical:{promedio_vm_corti}")
            
            
            
            
            # Aplica el cambio de volumen solo si la fracción de volumen óseo supera un umbral
            array_BVTV[condicion_trabecular] = array_BVTV[condicion_trabecular] 
            array_BVTV[condicion_cortical] = array_BVTV[condicion_cortical] + expression_array_dot_vb[condicion_cortical]
            
            array_vm[condicion_trabecular] = (escalar_vm * array_COPY_BVTV[condicion_trabecular] + escalar_vm * array_formacion_vb[condicion_trabecular])/array_BVTV[condicion_trabecular]
            array_vm[condicion_cortical] = (escalar_vm * array_COPY_BVTV[condicion_cortical] + escalar_vm*array_formacion_vb[condicion_cortical])/array_BVTV[condicion_cortical]
            
            array_alpha= array_vm*escalar_rho_mineral/(escalar_rho_mineral*array_vm + escalar_rho_org*escalar_v_org)


            #-------------------------------------------------------------------------------------------------------
            array_BVTV_mean=np.mean(array_BVTV[condicion_cortical])
            array_rho_cortical_gr_cm_3 = array_BVTV[condicion_cortical]*(1.41 +1.29*array_alpha[condicion_cortical]) 
            array_rho_tejido = (1.41 +1.29*array_alpha[condicion_cortical]) 
            promedios_rho_tejido[escalar_contador_total_dias]= np.mean(array_rho_tejido)
           
            
            array_modulo_E2_cortical = 84370 * (array_BVTV[condicion_cortical] ** 2.58) * (array_alpha[condicion_cortical] ** 2.74) *1e3
#            array_modulo_E_f[condicion_cortical]= 84370 * (array_BVTV[condicion_cortical] ** 2.58) * (array_alpha ** 2.74) *1e3
            array_modulo_E_f = 84370 * (array_BVTV ** 2.58) * (array_alpha ** 2.74) *1e3

            array_damage = 1-(array_modulo_E_f/array_modulo_E_0)
            array_damage_cortical = 1 - (escalar_modulo_E2_cortical/escalar_mean_modulo_cortical)
            
            promedios_damage[escalar_contador_total_dias]= np.mean(array_damage_cortical)
            promedios_modulo_elasticidad[escalar_contador_total_dias]= escalar_modulo_E2_cortical
            promedios_rho_cortical_gr_cm_3[escalar_contador_total_dias]= np.mean(array_rho_cortical_gr_cm_3)
            promedios_BVTV[escalar_contador_total_dias]=array_BVTV_mean 
            escalar_energy[escalar_contador_total_dias] = energy
            promedios_alpha[escalar_contador_total_dias]=np.mean(array_alpha)





    
            
            # ----------------------------------------------------------------------------------------------------------
            # Aca se modifica densidad 
            array_densidad_material = (1.41 + 1.29 * array_alpha) 
            array_densidad = array_densidad_material * array_BVTV
            
            fem_function_densidad.x.array[:] = array_densidad*1e-6
            fem_function_densidad_gr_cm3.x.array[:] = array_densidad
            fem_function_BVTV.x.array[:] = array_BVTV
            fem_function_modulo_E_f.x.array[:] = array_modulo_E_f
            fem_function_damage.x.array[:] = array_damage
            fem_function_alpha.x.array[:]= array_alpha
            fem_function_densidad_material.x.array[:] = array_densidad_material
            # ----------------------------------------------------------------------------------------------------------
            
            
            # ----------------------------------------------------------------------------------------------------------
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_6/fem_function_array_modulo.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_modulo_E_f , tmp_cycle)    
            
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_6/fem_function_array_damage.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_damage , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_2/fem_function_alpha.xdmf", "w") as file:
            #      file.write_mesh(mesh)
            #      file.write_function(fem_function_alpha , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_6/fem_function_array_rdot.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_array_rdot , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_6/fem_function_array_psi.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_array_psi , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_6/fem_function_array_error.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_array_error , tmp_cycle)    
    
            
            #-------------------------------------------------------------------------------------------------
            ##---------------------------------------------------------------------------------------------------
            ##---------------------------------------------------------------------------------------------------

            # Continúa con la parte de resolución de ecuaciones y condiciones de frontera...
            # Al parecer tambien hay un error en el calculo de mu no es + sino -
            mu = fem_function_modulo_E_f / (2.0 * (1.0 + 0.3))
            lame = fem_function_modulo_E_f * 0.3/ ((1.0 + 0.3) * (1.0 - 2.0 * 0.3)) 
            # revisar el valor de lamnda pues al parecer hay un error  
                                    
            f = ufl.as_vector((0, 0, -fem_function_densidad * 9810 ))
            V = functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
            u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
           
            def sigma(v):
                """Return an expression for the stress σ given a displacement field"""
                return 2.0 * mu * ufl.sym(grad(v)) + lame * ufl.tr(ufl.sym(grad(v))) * ufl.Identity(3)

            def strain(u):
                return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
            a = form(inner(sigma(u), grad(v)) * dx)
            L = form(inner(f, v) * dx)

            def invariants_principal(A):
                i1 = ufl.tr(A)
                i2 = (ufl.tr(A)**2 - ufl.tr(A * A)) / 2
                i3 = ufl.det(A)
                return i1, i2, i3

            #----------------------------------------------------------------------------
            # Asignacion de la condición frontera 08/02/24 - R-ninety
            #----------------------------------------------------------------------------
            def boundary_z(x):
                return np.logical_and(x[2] >= 36, x[2] <= 37)
            
            # Ubicar los grados de libertad en las caras con z entre 36 y 37
            dofs = dolfinx.fem.locate_dofs_geometrical(V, boundary_z)
            
            # Definir la condición de frontera como desplazamiento 0 en todas las direcciones
            u_D = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0, 0, 0)))
            
            # Aplicar la condición de frontera
            bc = dolfinx.fem.dirichletbc(u_D, dofs, V)
            #----------------------------------------------------------------------------
            
            # ## Assemble and solve
            #
            # The bilinear form `a` is assembled into a matrix `A`, with
            # modifications for the Dirichlet boundary conditions. The call
            # `A.assemble()` completes any parallel communication required to
            # compute the matrix.


            A = assemble_matrix(a, bcs=[bc])
            A.assemble()
            
            

            # Coordenadas de los puntos donde se aplicarán las cargas puntuales
            point_138 = np.array([87.7431, 20.6717, 224.6261],dtype=np.float64)
            point_140 = np.array([85.6372, 33.9296, 226.9851],dtype=np.float64)
            point_141 = np.array([77.7509, 23.4239, 216.3084],dtype=np.float64)
            point_143 = np.array([78.43, 17.2494, 209.5734],dtype=np.float64)
            point_148 = np.array([73.5285, 30.5294, 203.1597],dtype=np.float64)
            point_52 = np.array([124.2809, 54.4087,180.5598],dtype=np.float64)

            # Cargas en la dirección Z y X

            def marker_52(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_52[0], atol=atol)
                close_y = np.isclose(x[1], point_52[1], atol=atol)
                close_z = np.isclose(x[2], point_52[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_138(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_138[0], atol=atol)
                close_y = np.isclose(x[1], point_138[1], atol=atol)
                close_z = np.isclose(x[2], point_138[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_140(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_140[0], atol=atol)
                close_y = np.isclose(x[1], point_140[1], atol=atol)
                close_z = np.isclose(x[2], point_140[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)
            def marker_141(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_141[0], atol=atol)
                close_y = np.isclose(x[1], point_141[1], atol=atol)
                close_z = np.isclose(x[2], point_141[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)
            def marker_143(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_143[0], atol=atol)
                close_y = np.isclose(x[1], point_143[1], atol=atol)
                close_z = np.isclose(x[2], point_143[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_148(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_148[0], atol=atol)
                close_y = np.isclose(x[1], point_148[1], atol=atol)
                close_z = np.isclose(x[2], point_148[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)



            b = assemble_vector(L)

            # Localiza los grados de libertad en el vértice
            vertices_52 = locate_dofs_geometrical(V, marker_138)
            dofs_52_x = (vertices_52[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_52_z = (vertices_52[0])*3 + 2  # Grado de libertad en la dirección z


            vertices_138 = locate_dofs_geometrical(V, marker_138)
            dofs_138_x = (vertices_138[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_138_z = (vertices_138[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_140 = locate_dofs_geometrical(V, marker_140)
            dofs_140_x = (vertices_140[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_140_z = (vertices_140[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_141 = locate_dofs_geometrical(V, marker_141)
            dofs_141_x = (vertices_141[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_141_z = (vertices_141[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_143 = locate_dofs_geometrical(V, marker_143)
            dofs_143_x = (vertices_143[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_143_z = (vertices_143[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_148 = locate_dofs_geometrical(V, marker_148)
            dofs_148_x = (vertices_148[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_148_z = (vertices_148[0])*3 + 2  # Grado de libertad en la dirección z



            #print("El valor de los DOFs en X y en Z en la  carga:", dofs_138_x, dofs_138_z)
            # Magnitudes de las cargas puntuales
            P_x = 0  # Carga en la dirección y 
            P_z = -2317   # Carga en la dirección z
            P_z2= 713       # Carga del abductor en z
                # Aplicar las cargas puntuales
            with b.localForm() as loc_b:
                loc_b.setValues(dofs_138_x, P_x)
                loc_b.setValues(dofs_138_z, P_z)
                loc_b.setValues(dofs_140_x, P_x)
                loc_b.setValues(dofs_140_z, P_z) 
                loc_b.setValues(dofs_141_x, P_x)
                loc_b.setValues(dofs_141_z, P_z) 
                loc_b.setValues(dofs_143_x, P_x)
                loc_b.setValues(dofs_143_z, P_z) 
                loc_b.setValues(dofs_148_x, P_x)
                loc_b.setValues(dofs_148_z, P_z)
                loc_b.setValues(dofs_52_x, P_x)
                loc_b.setValues(dofs_52_z, P_z2)
            
            apply_lifting(b, [a], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
            set_bc(b, [bc])
    
    
            ns = build_nullspace(V)
            A.setNearNullSpace(ns)
            A.setOption(PETSc.Mat.Option.SPD, True)  # type: ignore
    
            
            # +
            # Set solver options
            opts = PETSc.Options()  # type: ignore
            opts["ksp_type"] = "cg"
            opts["ksp_rtol"] = 1.0e-8
            opts["pc_type"] = "gamg"
        
            # Use Chebyshev smoothing for multigrid
            opts["mg_levels_ksp_type"] = "chebyshev"
            opts["mg_levels_pc_type"] = "jacobi"
        
            # Improve estimate of eigenvalues for Chebyshev smoothing
            opts["mg_levels_ksp_chebyshev_esteig_steps"] = 10
        
            # Create PETSc Krylov solver and turn convergence monitoring on
            solver = PETSc.KSP().create(mesh.comm)  # type: ignore
            solver.setFromOptions()
    
            # Set matrix operator
            solver.setOperators(A)
            
            # +
            uh2_in = Function(V)
    
            solver.solve(b, uh2_in.vector)
    
            uh2_in.x.scatter_forward()
            
            # ## Post-processing
            # +
            # sigma_dev = sigma(uh_in) - (1 / 3) * ufl.tr(sigma(uh_in)) * ufl.Identity(len(uh_in))
            # sigma_vm = ufl.sqrt((3 / 2) * inner(sigma_dev, sigma_dev))
            # W = functionspace(mesh, ("Discontinuous Lagrange", 0))
            
            # sigma_vm_expr = Expression(sigma_vm, W.element.interpolation_points())
            # sigma_vm_h = Function(W,name="Von_misses")
            # sigma_vm_h.interpolate(sigma_vm_expr)

            
            # ----------------------------------------------------------------------------------

            
           
            U = 0.5 * ufl.inner(sigma(uh2_in), strain(uh2_in)) * ufl.dx
    
            # Ensamblar la energía de deformación ----<< 09/02/24 ---<<< OKß
            energy = fem.assemble_scalar(fem.form(U)) 
           # print("Energía de deformación total:", energy)
           # print("Sigma_bar:", sigma_bar_i)
            print("#-------------------------------------------------------------------------------------------------")
            print(f"6.{dday} - El valor promedio de daño:{promedios_damage[escalar_contador_total_dias]}")
            print(f"6.{dday} - El valor promedio de la densidad:{promedios_rho_cortical_gr_cm_3[escalar_contador_total_dias]}")
            print(f"6.{dday} - El valor promedio de la fracción de volumen:{promedios_BVTV[escalar_contador_total_dias]}")
            print(f"6.{dday} - El valor promedio del modulo de elasticidad:{promedios_modulo_elasticidad[escalar_contador_total_dias]}")
            print("#-------------------------------------------------------------------------------------------------")
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_6/fem_function_modulo.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_modulo_E_f , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_6/fem_function_densidad_material.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_densidad_material , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_6/fem_function_BVTV.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_BVTV , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_6/fem_function_array_modulo.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_modulo_E_f , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_6/fem_function_array_vm.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_vm , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_6/fem_function_array_damage.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_damage , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_6/fem_function_alpha.xdmf", "w") as file:
         file.write_mesh(mesh)
         file.write_function(fem_function_alpha , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_6/fem_function_array_rdot.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_rdot , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_6/fem_function_array_psi.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_psi , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_6/fem_function_array_error.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_error , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_6/fem_function_densidad_fase_6.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_densidad_gr_cm3 , tmp_day)
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_6/fem_function_BVTV_fase_6.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_BVTV , tmp_day)
#-------------------------------------------------------------------------------------------
    # Para fase #7 Esta fase dura 32 dias + TR + TI
#-------------------------------------------------------------------------------------------

    print("---------------------------------------------------------")              
    print("6.El valor promedio de la BVTV es:", promedios_BVTV[escalar_contador_total_dias])
    print(f"Inicia la siete etapa -->> La energía: {energy}")
    print("---------------------------------------------------------")              
    tmp_lenght = escalar_contador_total_dias
    print(f"06-longitud del CSV-->>>>:{tmp_lenght}")
    # Para fase #7: Esta fase dura 100 días + TR + TI + TF
    for dday in range(1, 65):  # Ciclo externo con dtime va de 1 a 96
        t_prima_100_TR_TI_TF = 64  - dday
        escalar_contador_total_dias += 1
        escalar_contador_total_dias_modelo += 1 
        t_alpha = 57 + dday

        for cycle in range(1000, 10001, 1000):  # Ciclo interno va de 1000 en 1000 hasta 10000
            tmp_day = dday
            tmp_cycle = cycle / 1000
    
            # Calcula el índice de daño basado en la energía y el módulo de elasticidad
            expression_array_ei = ((2 * energy) / array_modulo_E_f) ** (1 / 2)
            array_psi = (expression_array_ei * cycle) ** (1 / 4)
            fem_function_array_psi.x.array[:]= array_psi
            # ---------------------------------------------------------------------------------------------------------
            # Cálculo del error.
            # ---------------------------------------------------------------------------------------------------------
            array_error= ((1/array_BVTV)**2)*array_psi - 50
            fem_function_array_error.x.array[:]= array_error

            # ---------------------------------------------------------------------------------------------------------
            # Cálculo del balance focal oseo
            # con el valor de balance focal puedo calcular el valor remodelacion trabecular.
            # ---------------------------------------------------------------------------------------------------------
            array_rdot = np.where(array_error > width,
                      ratecon4 * array_error + (ratecon3 - ratecon4) * width,
                      np.where((array_error <= width) & (array_error >= 0),
                               ratecon3 * array_error,
                               np.where((array_error < 0) & (array_error >= -width),
                                        ratecon2 * array_error,
                                        ratecon1 * array_error + (ratecon1 - ratecon2) * width)))
            fem_function_array_rdot.x.array[:]=array_rdot
            
    
            # Informa el valor de la energía
    
            # Calcula la señal inhibitoria basada en el índice de daño
            expression_array_s = (array_psi / (1 + array_psi)) * (1 - array_damage) ** (1)
            fem_function_array_s = fem.Function(W)
            fem_function_array_s.x.array[:] = expression_array_s
    
            # Determina la frecuencia de nacimiento de nuevas unidades basales
            array_for = escalar_fbio * (1 - expression_array_s)
    
            # Calcula la porosidad basada en la fracción de volumen óseo
            array_porosidad = 1 - array_BVTV
            array_porosity2 = array_porosidad * array_porosidad
            array_porosity3 = array_porosity2 * array_porosidad
            array_porosity4 = array_porosity3 * array_porosidad
            array_porosity5 = array_porosity4 * array_porosidad
            array_Sv = 0.03226 * array_porosidad - 0.09394 * array_porosity2 + 0.13396 * array_porosity3 - 0.10104 * array_porosity4 + 0.02876 * array_porosity5
            
            array_COPY_BVTV = array_BVTV
    
            # Calcula la variación de las unidades basales
            array_derivative_NBMU = array_Sv * array_rdot * escalar_V_bmu * escalar_contador_total_dias_modelo
            
            escalar_f2 = 0.8092 *(t_prima_100_TR_TI_TF)/escalar_TF 

            escalar_tmp_af7 = escalar_tmp_ar2*((escalar_f2)**2 -  0.0364) /escalar_TF
            expression_array_dot_vb = array_derivative_NBMU * escalar_tmp_af7 
            
            array_formacion_vb[condicion_trabecular]= array_formacion_vb[condicion_trabecular] + array_derivative_NBMU[condicion_trabecular]/escalar_TF
            # en la etapa 5 no hay formacion hay reabsortion
            array_formacion_vb[condicion_cortical] = array_formacion_vb[condicion_cortical] - expression_array_dot_vb[condicion_cortical] 
      
            
            escalar_vm = escalar_vm_max - (escalar_vm_max -escalar_vm_prim)*math.exp(-escalar_km*(t_alpha -escalar_t_prim -escalar_t_nm))
           
            promedio_vm_trabe = np.mean(array_vm[condicion_trabecular]) 
            promedio_vm_corti = np.mean(array_vm[condicion_cortical])
            
            print(f"vm_trabecular:{promedio_vm_trabe}")
            print(f"vm_cortical:{promedio_vm_corti}")
            
            
            
            
            
            # Aplica el cambio de volumen solo si la fracción de volumen óseo supera un umbral
            array_BVTV[condicion_trabecular] = array_BVTV[condicion_trabecular] + array_derivative_NBMU[condicion_trabecular]/escalar_TF
            array_BVTV[condicion_cortical] =    array_BVTV[condicion_cortical] - expression_array_dot_vb[condicion_cortical]
            
            array_vm[condicion_trabecular] = (escalar_vm * array_COPY_BVTV[condicion_trabecular] + escalar_vm * array_formacion_vb[condicion_trabecular])/array_BVTV[condicion_trabecular]
            array_vm[condicion_cortical] = (escalar_vm * array_COPY_BVTV[condicion_cortical] + escalar_vm*array_formacion_vb[condicion_cortical])/array_BVTV[condicion_cortical]
            
            array_alpha= array_vm*escalar_rho_mineral/(escalar_rho_mineral*array_vm + escalar_rho_org*escalar_v_org)


    
            #-------------------------------------------------------------------------------------------------------
            array_BVTV_mean=np.mean(array_BVTV[condicion_cortical])
            array_rho_cortical_gr_cm_3 = array_BVTV[condicion_cortical]*(1.41 +1.29*array_alpha[condicion_cortical]) 
            array_rho_tejido = (1.41 +1.29*array_alpha[condicion_cortical]) 
            promedios_rho_tejido[escalar_contador_total_dias]= np.mean(array_rho_tejido)
            
            
            array_modulo_E2_cortical = 84370 * (array_BVTV[condicion_cortical] ** 2.58) * (array_alpha[condicion_cortical] ** 2.74) *1e3
            escalar_modulo_E2_cortical = np.mean(array_modulo_E2_cortical)
#            array_modulo_E_f[condicion_cortical]= 84370 * (array_BVTV[condicion_cortical] ** 2.58) * (array_alpha ** 2.74) *1e3
            array_modulo_E_f = 84370 * (array_BVTV ** 2.58) * (array_alpha ** 2.74) *1e3

            array_damage = 1-(array_modulo_E_f/array_modulo_E_0)
            array_damage_cortical = 1 - (escalar_modulo_E2_cortical/escalar_mean_modulo_cortical)


            promedios_damage[escalar_contador_total_dias]= np.mean(array_damage_cortical)
            promedios_modulo_elasticidad[escalar_contador_total_dias]= escalar_modulo_E2_cortical
            promedios_rho_cortical_gr_cm_3[escalar_contador_total_dias]= np.mean(array_rho_cortical_gr_cm_3)
            promedios_BVTV[escalar_contador_total_dias]=array_BVTV_mean 
            escalar_energy[escalar_contador_total_dias] = energy
            promedios_alpha[escalar_contador_total_dias]=np.mean(array_alpha)

    
            
            # ----------------------------------------------------------------------------------------------------------
            # Aca se modifica densidad 
            # ----------------------------------------------------------------------------------------------------------
            # Aca se modifica densidad 
            array_densidad_material = (1.41 + 1.29 * array_alpha) 
            array_densidad = array_densidad_material * array_BVTV

            fem_function_densidad.x.array[:] = array_densidad*1e-6
            fem_function_densidad_gr_cm3.x.array[:] = array_densidad
            fem_function_BVTV.x.array[:] = array_BVTV
            fem_function_modulo_E_f.x.array[:] = array_modulo_E_f
            fem_function_damage.x.array[:] = array_damage
            fem_function_alpha.x.array[:]= array_alpha
            fem_function_densidad_material.x.array[:] = array_densidad_material
            # ----------------------------------------------------------------------------------------------------------
            
            
            # ----------------------------------------------------------------------------------------------------------
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/fem_function_array_modulo.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_modulo_E_f , tmp_cycle)    
            
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/fem_function_array_damage.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_damage , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_2/fem_function_alpha.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_alpha , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/fem_function_array_rdot.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_array_rdot , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/fem_function_array_psi.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_array_psi , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/fem_function_array_error.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_array_error , tmp_cycle)    
    
   
            #-------------------------------------------------------------------------------------------------
            print("#-------------------------------------------------------------------------------------------------")
            print(f"7.{dday} - El valor promedio de daño:{promedios_damage[escalar_contador_total_dias]}")
            print(f"7.{dday} - El valor promedio de la densidad:{promedios_rho_cortical_gr_cm_3[escalar_contador_total_dias]}")
            print(f"7.{dday} - El valor promedio de la fracción de volumen:{promedios_BVTV[escalar_contador_total_dias]}")
            print(f"7.{dday} - El valor promedio del modulo de elasticidad:{promedios_modulo_elasticidad[escalar_contador_total_dias]}")
            print("#-------------------------------------------------------------------------------------------------")

    
            mu = fem_function_modulo_E_f / (2.0 * (1.0 + 0.3))
            # Al parecer tambien hay un error en el calculo de mu no es + sino -
            lame = fem_function_modulo_E_f * 0.3/ ((1.0 + 0.3) * (1.0 - 2.0 * 0.3)) 
             # revisar el valor de lamnda pues al parecer hay un error  
                                     
            f = ufl.as_vector((0, 0, -fem_function_densidad * 9810 ))
            V = functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
            u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
            
            def sigma(v):
                 """Return an expression for the stress σ given a displacement field"""
                 return 2.0 * mu * ufl.sym(grad(v)) + lame * ufl.tr(ufl.sym(grad(v))) * ufl.Identity(3)
            
            def strain(u):
                 return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
            a = form(inner(sigma(u), grad(v)) * dx)
            L = form(inner(f, v) * dx)
            
            def invariants_principal(A):
                 i1 = ufl.tr(A)
                 i2 = (ufl.tr(A)**2 - ufl.tr(A * A)) / 2
                 i3 = ufl.det(A)
                 return i1, i2, i3
            
             #----------------------------------------------------------------------------
             # Asignacion de la condición frontera 08/02/24 - R-ninety
             #----------------------------------------------------------------------------
            def boundary_z(x):
                 return np.logical_and(x[2] >= 36, x[2] <= 37)
             
             # Ubicar los grados de libertad en las caras con z entre 36 y 37
            dofs = dolfinx.fem.locate_dofs_geometrical(V, boundary_z)
             
             # Definir la condición de frontera como desplazamiento 0 en todas las direcciones
            u_D = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0, 0, 0)))
             
             # Aplicar la condición de frontera
            bc = dolfinx.fem.dirichletbc(u_D, dofs, V)
             #----------------------------------------------------------------------------
             
             # ## Assemble and solve
             #
             # The bilinear form `a` is assembled into a matrix `A`, with
             # modifications for the Dirichlet boundary conditions. The call
             # `A.assemble()` completes any parallel communication required to
             # compute the matrix.
            
            
            A = assemble_matrix(a, bcs=[bc])
            A.assemble()
            # Coordenadas de los puntos donde se aplicarán las cargas puntuales
            point_138 = np.array([87.7431, 20.6717, 224.6261],dtype=np.float64)
            point_140 = np.array([85.6372, 33.9296, 226.9851],dtype=np.float64)
            point_141 = np.array([77.7509, 23.4239, 216.3084],dtype=np.float64)
            point_143 = np.array([78.43, 17.2494, 209.5734],dtype=np.float64)
            point_148 = np.array([73.5285, 30.5294, 203.1597],dtype=np.float64)
            point_52 = np.array([124.2809, 54.4087,180.5598],dtype=np.float64)

            # Cargas en la dirección Z y X

            def marker_52(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_52[0], atol=atol)
                close_y = np.isclose(x[1], point_52[1], atol=atol)
                close_z = np.isclose(x[2], point_52[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_138(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_138[0], atol=atol)
                close_y = np.isclose(x[1], point_138[1], atol=atol)
                close_z = np.isclose(x[2], point_138[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_140(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_140[0], atol=atol)
                close_y = np.isclose(x[1], point_140[1], atol=atol)
                close_z = np.isclose(x[2], point_140[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)
            def marker_141(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_141[0], atol=atol)
                close_y = np.isclose(x[1], point_141[1], atol=atol)
                close_z = np.isclose(x[2], point_141[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)
            def marker_143(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_143[0], atol=atol)
                close_y = np.isclose(x[1], point_143[1], atol=atol)
                close_z = np.isclose(x[2], point_143[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_148(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_148[0], atol=atol)
                close_y = np.isclose(x[1], point_148[1], atol=atol)
                close_z = np.isclose(x[2], point_148[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)



            b = assemble_vector(L)

            # Localiza los grados de libertad en el vértice
            vertices_52 = locate_dofs_geometrical(V, marker_138)
            dofs_52_x = (vertices_52[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_52_z = (vertices_52[0])*3 + 2  # Grado de libertad en la dirección z


            vertices_138 = locate_dofs_geometrical(V, marker_138)
            dofs_138_x = (vertices_138[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_138_z = (vertices_138[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_140 = locate_dofs_geometrical(V, marker_140)
            dofs_140_x = (vertices_140[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_140_z = (vertices_140[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_141 = locate_dofs_geometrical(V, marker_141)
            dofs_141_x = (vertices_141[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_141_z = (vertices_141[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_143 = locate_dofs_geometrical(V, marker_143)
            dofs_143_x = (vertices_143[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_143_z = (vertices_143[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_148 = locate_dofs_geometrical(V, marker_148)
            dofs_148_x = (vertices_148[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_148_z = (vertices_148[0])*3 + 2  # Grado de libertad en la dirección z



            #print("El valor de los DOFs en X y en Z en la  carga:", dofs_138_x, dofs_138_z)
            # Magnitudes de las cargas puntuales
            P_x = 0  # Carga en la dirección y 
            P_z = -2317   # Carga en la dirección z
            P_z2= 713       # Carga del abductor en z
                # Aplicar las cargas puntuales
            with b.localForm() as loc_b:
                loc_b.setValues(dofs_138_x, P_x)
                loc_b.setValues(dofs_138_z, P_z)
                loc_b.setValues(dofs_140_x, P_x)
                loc_b.setValues(dofs_140_z, P_z) 
                loc_b.setValues(dofs_141_x, P_x)
                loc_b.setValues(dofs_141_z, P_z) 
                loc_b.setValues(dofs_143_x, P_x)
                loc_b.setValues(dofs_143_z, P_z) 
                loc_b.setValues(dofs_148_x, P_x)
                loc_b.setValues(dofs_148_z, P_z)
                loc_b.setValues(dofs_52_x, P_x)
                loc_b.setValues(dofs_52_z, P_z2)
            
             
            
             
             
            apply_lifting(b, [a], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
            set_bc(b, [bc])
            
            
            ns = build_nullspace(V)
            A.setNearNullSpace(ns)
            A.setOption(PETSc.Mat.Option.SPD, True)  # type: ignore
            
             
             # +
             # Set solver options
            opts = PETSc.Options()  # type: ignore
            opts["ksp_type"] = "cg"
            opts["ksp_rtol"] = 1.0e-8
            opts["pc_type"] = "gamg"
            
             # Use Chebyshev smoothing for multigrid
            opts["mg_levels_ksp_type"] = "chebyshev"
            opts["mg_levels_pc_type"] = "jacobi"
            
             # Improve estimate of eigenvalues for Chebyshev smoothing
            opts["mg_levels_ksp_chebyshev_esteig_steps"] = 10
            
             # Create PETSc Krylov solver and turn convergence monitoring on
            solver = PETSc.KSP().create(mesh.comm)  # type: ignore
            solver.setFromOptions()
            
             # Set matrix operator
            solver.setOperators(A)
             
             # +
            uh_in = Function(V)
            
            solver.solve(b, uh_in.vector)
            
            uh_in.x.scatter_forward()
             
             # ## Post-processing
             # +
            # sigma_dev = sigma(uh_in) - (1 / 3) * ufl.tr(sigma(uh_in)) * ufl.Identity(len(uh_in))
            # sigma_vm = ufl.sqrt((3 / 2) * inner(sigma_dev, sigma_dev))
             
            # sigma_vm_expr = Expression(sigma_vm, W.element.interpolation_points())
            # sigma_vm_h.interpolate(sigma_vm_expr)
            
            #valor_sigma = sigma(uh_in)
            # sigma_value_expression =  Expression(valor_sigma , W.element.interpolation_points())
            # value_sigma_h.interpolate(sigma_value_expression)
             
             # ----------------------------------------------------------------------------------
            
                
             
            
            U = 0.5 * ufl.inner(sigma(uh_in), strain(uh_in)) * ufl.dx
            
             # Ensamblar la energía de deformación ----<< 09/02/24 ---<<< OKß
            energy = fem.assemble_scalar(fem.form(U)) 
            
            # Archivo para almacenar la densidad al final de la fase #7
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/misses_.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(sigma_vm_h , dday)
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/displacements_.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(uh_in, dday)
    # with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/sigma_.xdmf", "w") as file:
    #     file.write_mesh(mesh)
    #     file.write_function(valor_sigma , dday)
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/fem_function_modulo.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_modulo_E_f , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/fem_function_densidad_material.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_densidad_material , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/fem_function_BVTV.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_BVTV , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/fem_function_array_modulo.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_modulo_E_f , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/fem_function_array_vm.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_vm , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/fem_function_array_damage.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_damage , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/fem_function_alpha.xdmf", "w") as file:
         file.write_mesh(mesh)
         file.write_function(fem_function_alpha , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/fem_function_array_rdot.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_rdot , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/fem_function_array_psi.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_psi , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/fem_function_array_error.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_error , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/fem_function_densidad_fase_7.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_densidad_gr_cm3 , tmp_day)
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/fem_function_BVTV_fase_7.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_BVTV , tmp_day)
        
#   -------------------------------------------------------------------------------------------------------------
#      Segundo ciclo de remodelación  
#   -------------------------------------------------------------------------------------------------------------
        
    for dday in range(1, 5):  # Ciclo externo con dtime va de 1 a 24
       
        escalar_contador_total_dias += 1
        escalar_contador_total_dias_modelo +=1
        t_prima_24 = 24 - dday
        tmp_day = dday
        t_alpha = 121 + dday

        for cycle in range(1000, 10001, 1000):  # Ciclo interno va de 1000 en 1000 hasta 10000
            tmp_day = dday
            tmp_cycle = cycle / 1000
            print(f"energy {dday}:{energy} - cycle:{cycle}")

            expression_array_ei = ((2 * energy) / array_modulo_E_f)**(1/2)
            array_psi = (expression_array_ei * cycle)**(1/4)
            fem_function_array_psi.x.array[:]= array_psi
            # ---------------------------------------------------------------------------------------------------------
            # Cálculo del error.
            # ---------------------------------------------------------------------------------------------------------
            array_error= ((1/array_BVTV)**2)*array_psi - 50
            fem_function_array_error.x.array[:]= array_error

            # ---------------------------------------------------------------------------------------------------------
            # Cálculo del balance focal oseo
            # con el valor de balance focal puedo calcular el valor remodelacion trabecular.
            # ---------------------------------------------------------------------------------------------------------
            array_rdot = np.where(array_error > width,
                      ratecon4 * array_error + (ratecon3 - ratecon4) * width,
                      np.where((array_error <= width) & (array_error >= 0),
                               ratecon3 * array_error,
                               np.where((array_error < 0) & (array_error >= -width),
                                        ratecon2 * array_error,
                                        ratecon1 * array_error + (ratecon1 - ratecon2) * width)))
            fem_function_array_rdot.x.array[:]=array_rdot
            
            # ----------------------------------------------------------------------------------------------------------
            # Paso 4: Calcular el valor de la señal inhibitoria.
            # ----------------------------------------------------------------------------------------------------------
           
            # ---------------------------------------------------------------------------------------------------- 
            # El cálculo de la porosidad proviene del calculo del valor de la superficie específica
            array_porosidad = 1 - array_BVTV 
            array_porosity2 = array_porosidad * array_porosidad
            array_porosity3 = array_porosity2 * array_porosidad
            array_porosity4 = array_porosity3 * array_porosidad
            array_porosity5 = array_porosity4 * array_porosidad
            array_Sv = 0.03226 * array_porosidad - 0.09394 * array_porosity2 + 0.13396 * array_porosity3 - 0.10104 * array_porosity4 + 0.02876 * array_porosity5 
            # --------------------------------------------------------------------------------------
            # El calculo de la variacion de la variacion de las BMUs
            array_derivative_NBMU = array_Sv * array_rdot * escalar_V_bmu * escalar_contador_total_dias_modelo

            expression_array_dot_vr = (array_BVTV) * array_derivative_NBMU * escalar_tmp_ar1 * (t_prima_24 /escalar_TR)**2 
            fem_function_vr_dot.x.array[:] = expression_array_dot_vr
            
            #array_formacion_vb[condicion_trabecular]= array_formacion_vb[condicion_trabecular] - array_derivative_NBMU[condicion_trabecular]/escalar_TR
            # en la etapa 5 no hay formacion hay reabsortion
            #array_formacion_vb[condicion_cortical] = array_formacion_vb[condicion_cortical] + expression_array_dot_vb[condicion_cortical] /escalar_TR
      
            
            escalar_vm = escalar_vm_max - (escalar_vm_max -escalar_vm_prim)*math.exp(-escalar_km*(t_alpha -escalar_t_prim -escalar_t_nm))
           
            promedio_vm_trabe = np.mean(array_vm[condicion_trabecular]) 
            promedio_vm_corti = np.mean(array_vm[condicion_cortical])
            
            print(f"vm_trabecular:{promedio_vm_trabe}")
            print(f"vm_cortical:{promedio_vm_corti}")
            
          
            array_BVTV[condicion_trabecular] = array_BVTV[condicion_trabecular] - array_derivative_NBMU[condicion_trabecular] / escalar_TR
            array_BVTV[condicion_cortical] = array_BVTV[condicion_cortical] + expression_array_dot_vr[condicion_cortical] / escalar_TR
            
            array_vm[condicion_trabecular] = (escalar_vm * array_COPY_BVTV[condicion_trabecular] + escalar_vm * array_formacion_vb[condicion_trabecular])/array_BVTV[condicion_trabecular]
            array_vm[condicion_cortical] = (escalar_vm * array_COPY_BVTV[condicion_cortical] + escalar_vm*array_formacion_vb[condicion_cortical])/array_BVTV[condicion_cortical]
            
            array_alpha= array_vm*escalar_rho_mineral/(escalar_rho_mineral*array_vm + escalar_rho_org*escalar_v_org)

            #array_BVTV = array_BVTV + expression_array_dot_vr

            
            #---------------------------------------------------------------------------
            array_BVTV_mean=np.mean(array_BVTV[condicion_cortical])
            array_rho_cortical_gr_cm_3 = array_BVTV[condicion_cortical]*(1.41 +1.29*array_alpha[condicion_cortical]) 
            array_rho_tejido = (1.41 +1.29*array_alpha[condicion_cortical]) 
            promedios_rho_tejido[escalar_contador_total_dias ]= np.mean(array_rho_tejido)
            
            array_modulo_E2_cortical = 84370 * (array_BVTV[condicion_cortical] ** 2.58) * (array_alpha[condicion_cortical] ** 2.74) *1e3
            escalar_modulo_E2_cortical = np.mean(array_modulo_E2_cortical)
            # array_modulo_E_f[condicion_cortical]= 84370 * (array_BVTV[condicion_cortical] ** 2.58) * (array_alpha ** 2.74) *1e3
            array_modulo_E_f= 84370 * (array_BVTV ** 2.58) * (array_alpha ** 2.74) *1e3
            
            array_damage = 1-(array_modulo_E_f/array_modulo_E_0)
            
            array_damage_cortical = 1 - (escalar_modulo_E2_cortical/escalar_mean_modulo_cortical)
            # ----------------------------------------------------------------------------------------------------------
            # Los promedios de salida
            # ----------------------------------------------------------------------------------------------------------
            
            promedios_damage[escalar_contador_total_dias]= np.mean(array_damage_cortical)
            promedios_modulo_elasticidad[escalar_contador_total_dias]= escalar_modulo_E2_cortical
            promedios_rho_cortical_gr_cm_3[escalar_contador_total_dias]= np.mean(array_rho_cortical_gr_cm_3)
            promedios_BVTV[escalar_contador_total_dias]=array_BVTV_mean 
            escalar_energy[escalar_contador_total_dias] = energy
            promedios_alpha[escalar_contador_total_dias]=np.mean(array_alpha)

            # ----------------------------------------------------------------------------------------------------------
            # Aca se modifica densidad 
            # ----------------------------------------------------------------------------------------------------------
            # Aca se modifica densidad 
            array_densidad_material = (1.41 + 1.29 * array_alpha) 
            array_densidad = array_densidad_material * array_BVTV
            
            fem_function_densidad.x.array[:] = array_densidad*1e-6
            fem_function_densidad_gr_cm3.x.array[:] = array_densidad
            fem_function_BVTV.x.array[:] = array_BVTV
            fem_function_modulo_E_f.x.array[:] = array_modulo_E_f
            fem_function_damage.x.array[:] = array_damage
            fem_function_alpha.x.array[:] = array_alpha
            fem_function_densidad_material.x.array[:] = array_densidad_material

            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_1/fem_function_array_modulo.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_modulo_E_f , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_1/fem_function_array_rdot.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_array_rdot , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_1/fem_function_array_psi.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_array_psi , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_1/fem_function_array_error.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_array_error , tmp_cycle)    
            # with XDMFFile(mesh.comm, "./20_01_05_24/fase_1/fem_function_array_alpha.xdmf", "w") as file:
            #     file.write_mesh(mesh)
            #     file.write_function(fem_function_alpha , tmp_cycle)    
            # # --------------------------------------------------------------------------------------------------
            
            # ##---------------------------------------------------------------------------------------------------
            # ##---------------------------------------------------------------------------------------------------

            # Continúa con la parte de resolución de ecuaciones y condiciones de frontera...
            # Al parecer tambien hay un error en el calculo de mu no es + sino -
            mu = fem_function_modulo_E_f / (2.0 * (1.0 + 0.3))
            lame = fem_function_modulo_E_f * 0.3/ ((1.0 + 0.3) * (1.0 - 2.0 * 0.3)) 
            # revisar el valor de lamnda pues al parecer hay un error  
                                    
            f = ufl.as_vector((0, 0, -fem_function_densidad * 9810 ))
            V = functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
            u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
           
            def sigma(v):
                """Return an expression for the stress σ given a displacement field"""
                return 2.0 * mu * ufl.sym(grad(v)) + lame * ufl.tr(ufl.sym(grad(v))) * ufl.Identity(3)

            def strain(u):
                return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
            a = form(inner(sigma(u), grad(v)) * dx)
            L = form(inner(f, v) * dx)

            def invariants_principal(A):
                i1 = ufl.tr(A)
                i2 = (ufl.tr(A)**2 - ufl.tr(A * A)) / 2
                i3 = ufl.det(A)
                return i1, i2, i3

            #----------------------------------------------------------------------------
            # Asignacion de la condición frontera 08/02/24 - R-ninety
            #----------------------------------------------------------------------------
            def boundary_z(x):
                return np.logical_and(x[2] >= 36, x[2] <= 37)
            
            # Ubicar los grados de libertad en las caras con z entre 36 y 37
            dofs = dolfinx.fem.locate_dofs_geometrical(V, boundary_z)
            
            # Definir la condición de frontera como desplazamiento 0 en todas las direcciones
            u_D = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0, 0, 0)))
            
            # Aplicar la condición de frontera
            bc = dolfinx.fem.dirichletbc(u_D, dofs, V)
            #----------------------------------------------------------------------------
            
            # ## Assemble and solve
            #
            # The bilinear form `a` is assembled into a matrix `A`, with
            # modifications for the Dirichlet boundary conditions. The call
            # `A.assemble()` completes any parallel communication required to
            # compute the matrix.


            A = assemble_matrix(a, bcs=[bc])
            A.assemble()
            
            # Coordenadas de los puntos donde se aplicarán las cargas puntuales
            point_138 = np.array([87.7431, 20.6717, 224.6261],dtype=np.float64)
            point_140 = np.array([85.6372, 33.9296, 226.9851],dtype=np.float64)
            point_141 = np.array([77.7509, 23.4239, 216.3084],dtype=np.float64)
            point_143 = np.array([78.43, 17.2494, 209.5734],dtype=np.float64)
            point_148 = np.array([73.5285, 30.5294, 203.1597],dtype=np.float64)
            point_52 = np.array([124.2809, 54.4087,180.5598],dtype=np.float64)

            # Cargas en la dirección Z y X

            def marker_52(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_52[0], atol=atol)
                close_y = np.isclose(x[1], point_52[1], atol=atol)
                close_z = np.isclose(x[2], point_52[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_138(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_138[0], atol=atol)
                close_y = np.isclose(x[1], point_138[1], atol=atol)
                close_z = np.isclose(x[2], point_138[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_140(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_140[0], atol=atol)
                close_y = np.isclose(x[1], point_140[1], atol=atol)
                close_z = np.isclose(x[2], point_140[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)
            def marker_141(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_141[0], atol=atol)
                close_y = np.isclose(x[1], point_141[1], atol=atol)
                close_z = np.isclose(x[2], point_141[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)
            def marker_143(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_143[0], atol=atol)
                close_y = np.isclose(x[1], point_143[1], atol=atol)
                close_z = np.isclose(x[2], point_143[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)

            def marker_148(x):
                # Asegurarse de que x se compara correctamente con las coordenadas de point_138
                close_x = np.isclose(x[0], point_148[0], atol=atol)
                close_y = np.isclose(x[1], point_148[1], atol=atol)
                close_z = np.isclose(x[2], point_148[2], atol=atol)
                return np.logical_and(np.logical_and(close_x, close_y), close_z)



            b = assemble_vector(L)

            # Localiza los grados de libertad en el vértice
            vertices_52 = locate_dofs_geometrical(V, marker_138)
            dofs_52_x = (vertices_52[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_52_z = (vertices_52[0])*3 + 2  # Grado de libertad en la dirección z


            vertices_138 = locate_dofs_geometrical(V, marker_138)
            dofs_138_x = (vertices_138[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_138_z = (vertices_138[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_140 = locate_dofs_geometrical(V, marker_140)
            dofs_140_x = (vertices_140[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_140_z = (vertices_140[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_141 = locate_dofs_geometrical(V, marker_141)
            dofs_141_x = (vertices_141[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_141_z = (vertices_141[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_143 = locate_dofs_geometrical(V, marker_143)
            dofs_143_x = (vertices_143[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_143_z = (vertices_143[0])*3 + 2  # Grado de libertad en la dirección z

            vertices_148 = locate_dofs_geometrical(V, marker_148)
            dofs_148_x = (vertices_148[0])*3 + 0  # Grado de libertad en la dirección y
            dofs_148_z = (vertices_148[0])*3 + 2  # Grado de libertad en la dirección z



            #print("El valor de los DOFs en X y en Z en la  carga:", dofs_138_x, dofs_138_z)
            # Magnitudes de las cargas puntuales
            P_x = 0  # Carga en la dirección y 
            P_z = -2317   # Carga en la dirección z
            P_z2= 713       # Carga del abductor en z
                # Aplicar las cargas puntuales
            with b.localForm() as loc_b:
                loc_b.setValues(dofs_138_x, P_x)
                loc_b.setValues(dofs_138_z, P_z)
                loc_b.setValues(dofs_140_x, P_x)
                loc_b.setValues(dofs_140_z, P_z) 
                loc_b.setValues(dofs_141_x, P_x)
                loc_b.setValues(dofs_141_z, P_z) 
                loc_b.setValues(dofs_143_x, P_x)
                loc_b.setValues(dofs_143_z, P_z) 
                loc_b.setValues(dofs_148_x, P_x)
                loc_b.setValues(dofs_148_z, P_z)
                loc_b.setValues(dofs_52_x, P_x)
                loc_b.setValues(dofs_52_z, P_z2)

            
            apply_lifting(b, [a], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
            set_bc(b, [bc])
    
    
            ns = build_nullspace(V)
            A.setNearNullSpace(ns)
            A.setOption(PETSc.Mat.Option.SPD, True)  # type: ignore
    
            
            # +
            # Set solver options
            opts = PETSc.Options()  # type: ignore
            opts["ksp_type"] = "cg"
            opts["ksp_rtol"] = 1.0e-8
            opts["pc_type"] = "gamg"
        
            # Use Chebyshev smoothing for multigrid
            opts["mg_levels_ksp_type"] = "chebyshev"
            opts["mg_levels_pc_type"] = "jacobi"
        
            # Improve estimate of eigenvalues for Chebyshev smoothing
            opts["mg_levels_ksp_chebyshev_esteig_steps"] = 10
        
            # Create PETSc Krylov solver and turn convergence monitoring on
            solver = PETSc.KSP().create(mesh.comm)  # type: ignore
            solver.setFromOptions()
    
            # Set matrix operator
            solver.setOperators(A)
            
            # +
            uh2_in = Function(V)
    
            solver.solve(b, uh2_in.vector)
    
            uh2_in.x.scatter_forward()
            
            # ## Post-processing
            # +
            sigma_dev = sigma(uh2_in) - (1 / 3) * ufl.tr(sigma(uh2_in)) * ufl.Identity(len(uh2_in))
            sigma_vm = ufl.sqrt((3 / 2) * inner(sigma_dev, sigma_dev))
            W = functionspace(mesh, ("Discontinuous Lagrange", 0))
            
            sigma_vm_expr = Expression(sigma_vm, W.element.interpolation_points())
            sigma_vm_h = Function(W,name="Von_misses")
            sigma_vm_h.interpolate(sigma_vm_expr)

            
            # ----------------------------------------------------------------------------------

            
           
            U=0.5*ufl.inner(sigma(uh2_in),strain(uh2_in))*ufl.dx
    
            # Ensamblar la energía de deformación ----<< 09/02/24 ---<<< OKß
            energy = fem.assemble_scalar(fem.form(U)) 
           # print("Energía de deformación total:", energy)
           # print("Sigma_bar:", sigma_bar_i)
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_8/misses_.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(sigma_vm_h , dday)
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_8/displacements_.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(uh_in, dday)
    # with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/sigma_.xdmf", "w") as file:
    #     file.write_mesh(mesh)
    #     file.write_function(valor_sigma , dday)
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_8/fem_function_modulo.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_modulo_E_f , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_8/fem_function_densidad_material.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_densidad_material , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_8/fem_function_BVTV.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_BVTV , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_8/fem_function_array_modulo.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_modulo_E_f , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_8/fem_function_array_vm.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_vm , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_8/fem_function_array_damage.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_damage , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_8/fem_function_alpha.xdmf", "w") as file:
         file.write_mesh(mesh)
         file.write_function(fem_function_alpha , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_8/fem_function_array_rdot.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_rdot , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_8/fem_function_array_psi.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_psi , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_8/fem_function_array_error.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_array_error , tmp_cycle)    
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_8/fem_function_densidad_fase_7.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_densidad_gr_cm3 , tmp_day)
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_8/fem_function_BVTV_fase_8.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_BVTV , tmp_day)
            
            
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_8/fem_function_densidad_fase_1.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_densidad_gr_cm3 , tmp_day)
    with XDMFFile(mesh.comm, "./20_01_05_24/fase_8/fem_function_BVTV_fase_1.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fem_function_BVTV , tmp_day)
    print(f"El valor promedio de daño:{promedios_damage[escalar_contador_total_dias]}")
    print(f"El valor promedio de la densidad:{promedios_rho_cortical_gr_cm_3[escalar_contador_total_dias]}")
    print(f"El valor promedio de la fracción de volumen:{promedios_BVTV[escalar_contador_total_dias]}")
    print(f"1.{dday} - El valor promedio del modulo de elasticidad:{promedios_modulo_elasticidad[escalar_contador_total_dias]}")
    print(f" Finalizo la primera etapa -->> La energía: {escalar_energy[escalar_contador_total_dias]}")
  
# ----------------------------------------------------------------------------------------------------
# aca inicia fase 2.2 segunada secuencia en el modelo 
# ----------------------------------------------------------------------------------------------------



# #---------------------------------------------------------------------------------------------------
# # Para fase #2.2  el tiempo es menor de 8 días
# # expression_array_dot_vr = np.zeros(W.dofmap.index_map.size_local)
# # Phase 2            # Only  resorption aca deberia solamente  r hasta el día 32---->>>>>
# #   for TR in range(1, 1, 24):  # Ciclo interno va de 1 en 1 hasta 24 
# #---------------------------------------------------------------------------------------------------
#     escalar_contador_total_dias_modelo = 0
#     promedio_BVTV= np.mean(array_BVTV)
#     print("---------------------------------------------------------")              
#     print("El valor promedio de la BVTV es:", promedios_BVTV[escalar_contador_total_dias])
#     print(f" Inicia la segunda etapa -->> La energía: {energy}")
#     print("---------------------------------------------------------")
#     tmp_lenght = escalar_contador_total_dias
#     print(f"01-longitud del CSV-->>>>:{tmp_lenght}")
#     for dday in range(1, 8):  # Ciclo externo con dtime va de 1 a 8
        
#         tmp_day = dday
#         escalar_contador_total_dias += 1
#         escalar_contador_total_dias_modelo += 1
#         t_prima_8 = 8 - dday
#         t_alpha = 145 + dday
    
#         for cycle in range(1000, 10001, 1000):  # Ciclo interno va de 1000 en 1000 hasta 10000
#             tmp_cycle = cycle / 1000
    
#             expression_array_ei = ((2 * energy) / array_modulo_E_f) ** (1 / 2)
#             array_psi = (expression_array_ei * cycle) ** (1 / 4)
#             fem_function_array_psi.x.array[:]= array_psi

#             # ---------------------------------------------------------------------------------------------------------
#             # Cálculo del error.
#             # ---------------------------------------------------------------------------------------------------------
#             array_error= ((1/array_BVTV)**2)*array_psi - 50
#             fem_function_array_error.x.array[:]= array_error

#             # ---------------------------------------------------------------------------------------------------------
#             # Cálculo del balance focal oseo
#             # con el valor de balance focal puedo calcular el valor remodelacion trabecular.
#             # ---------------------------------------------------------------------------------------------------------
#             array_rdot = np.where(array_error > width,
#                       ratecon4 * array_error + (ratecon3 - ratecon4) * width,
#                       np.where((array_error <= width) & (array_error >= 0),
#                                ratecon3 * array_error,
#                                np.where((array_error < 0) & (array_error >= -width),
#                                         ratecon2 * array_error,
#                                         ratecon1 * array_error + (ratecon1 - ratecon2) * width)))
#             fem_function_array_rdot.x.array[:]=array_rdot
            

#             array_porosidad = 1 - array_BVTV
#             array_porosity2 = array_porosidad * array_porosidad
#             array_porosity3 = array_porosity2 * array_porosidad
#             array_porosity4 = array_porosity3 * array_porosidad
#             array_porosity5 = array_porosity4 * array_porosidad
#             array_Sv = 0.03226 * array_porosidad - 0.09394 * array_porosity2 + 0.13396 * array_porosity3 - 0.10104 * array_porosity4 + 0.02876 * array_porosity5
    
#             array_derivative_NBMU = array_Sv * array_rdot *escalar_V_bmu * escalar_contador_total_dias_modelo
#             # ------------------------------------------------------------------------------------------------
#             # ------------------------------------------------------------------------------------------------
#             # Aca se esta modificando la fraccion de volumen de hueso cortical                        
#             # ------------------------------------------------------------------------------------------------

#             expression_array_dot_vr = array_BVTV * array_derivative_NBMU * escalar_tmp_ar1 / escalar_TR
#             fem_function_vr_dot.x.array[:] = expression_array_dot_vr
    
#             array_BVTV[condicion_trabecular] = array_BVTV[condicion_trabecular] - array_derivative_NBMU[condicion_trabecular] /escalar_TR
#             array_BVTV[condicion_cortical] = array_BVTV[condicion_cortical] + expression_array_dot_vr[condicion_cortical]
            
            
#             escalar_vm = escalar_vm_max - (escalar_vm_max -escalar_vm_prim)*math.exp(-escalar_km*(t_alpha -escalar_t_prim -escalar_t_nm))
           
#             promedio_vm_trabe = np.mean(array_vm[condicion_trabecular]) 
#             promedio_vm_corti = np.mean(array_vm[condicion_cortical])
            
#             print(f"vm_trabecular:{promedio_vm_trabe}")
#             print(f"vm_cortical:{promedio_vm_corti}")
            
          
#             array_BVTV[condicion_trabecular] = array_BVTV[condicion_trabecular] - array_derivative_NBMU[condicion_trabecular] / escalar_TR
#             array_BVTV[condicion_cortical] = array_BVTV[condicion_cortical] + expression_array_dot_vr[condicion_cortical] / escalar_TR
            
#             array_vm[condicion_trabecular] = (escalar_vm * array_COPY_BVTV[condicion_trabecular] + escalar_vm * array_formacion_vb[condicion_trabecular])/array_BVTV[condicion_trabecular]
#             array_vm[condicion_cortical] = (escalar_vm * array_COPY_BVTV[condicion_cortical] + escalar_vm*array_formacion_vb[condicion_cortical])/array_BVTV[condicion_cortical]
            
#             array_alpha= array_vm*escalar_rho_mineral/(escalar_rho_mineral*array_vm + escalar_rho_org*escalar_v_org)

            
            
            
#             #+ expression_array_dot_vr[condicion_cortical] * 0.00034 / array_BVTV[condicion_cortical]
#             #-------------------------------------------------------------------------------------------------------
#             array_BVTV_mean=np.mean(array_BVTV[condicion_cortical])
#             array_rho_cortical_gr_cm_3 = array_BVTV[condicion_cortical]*(1.41 +1.29*array_alpha[condicion_cortical]) 
#             array_rho_tejido = (1.41 +1.29*array_alpha[condicion_cortical]) 
#             promedios_rho_tejido[escalar_contador_total_dias]= np.mean(array_rho_tejido)
            
            
#             array_modulo_E2_cortical = 84370 * (array_BVTV[condicion_cortical] ** 2.58) * (array_alpha[condicion_cortical] ** 2.74) *1e3
#             escalar_modulo_E2_cortical = np.mean(array_modulo_E2_cortical)
#             # array_modulo_E_f[condicion_cortical]= 84370 * (array_BVTV[condicion_cortical] ** 2.58) * (array_alpha ** 2.74) *1e3
#             array_modulo_E_f = 84370 * (array_BVTV ** 2.58) * (array_alpha ** 2.74) *1e3
            
#             array_damage = 1-(array_modulo_E_f/array_modulo_E_0)
#             array_damage_cortical = 1 - (escalar_modulo_E2_cortical/escalar_mean_modulo_cortical)
            
#             #-------------------------------------------------------------------------------------------------------            
#             promedios_damage[escalar_contador_total_dias]= np.mean(array_damage_cortical)
#             promedios_modulo_elasticidad[escalar_contador_total_dias]= escalar_modulo_E2_cortical
#             promedios_rho_cortical_gr_cm_3[escalar_contador_total_dias]= np.mean(array_rho_cortical_gr_cm_3)
#             promedios_BVTV[escalar_contador_total_dias]=array_BVTV_mean 
#             escalar_energy[escalar_contador_total_dias] = energy
#             promedios_alpha[escalar_contador_total_dias]=np.mean(array_alpha)


    
            
#             # ----------------------------------------------------------------------------------------------------------
#             # Aca se modifica densidad 
#             # ----------------------------------------------------------------------------------------------------------
#             #array_densidad[condicion_cortical] = (1.41 + 1.29 * alpha_ini) * array_BVTV[condicion_cortical]
#             # Aca se modifica densidad 
#             array_densidad_material = (1.41 + 1.29 * array_alpha) 
#             array_densidad = array_densidad_material * array_BVTV

#             fem_function_densidad_gr_cm3.x.array[:] = array_densidad
#             fem_function_densidad.x.array[:] = array_densidad*1e-6
#             fem_function_BVTV.x.array[:] = array_BVTV
#             fem_function_modulo_E_f.x.array[:] = array_modulo_E_f
#             fem_function_damage.x.array[:] = array_damage
#             fem_function_densidad_material.x.array[:] = array_densidad_material
            
#             # # ----------------------------------------------------------------------------------------------------------
#             # with XDMFFile(mesh.comm, "./20_01_05_24/fase_2/fem_function_array_modulo.xdmf", "w") as file:
#             #     file.write_mesh(mesh)
#             #     file.write_function(fem_function_modulo_E_f , tmp_cycle)    
#             # with XDMFFile(mesh.comm, "./20_01_05_24/fase_2/fem_function_array_damage.xdmf", "w") as file:
#             #     file.write_mesh(mesh)
#             #     file.write_function(fem_function_damage , tmp_cycle)    
#             # with XDMFFile(mesh.comm, "./20_01_05_24/fase_2/fem_function_array_rdot.xdmf", "w") as file:
#             #     file.write_mesh(mesh)
#             #     file.write_function(fem_function_array_rdot , tmp_cycle)    
#             # with XDMFFile(mesh.comm, "./20_01_05_24/fase_2/fem_function_array_psi.xdmf", "w") as file:
#             #     file.write_mesh(mesh)
#             #     file.write_function(fem_function_array_psi , tmp_cycle)    
#             # with XDMFFile(mesh.comm, "./20_01_05_24/fase_2/fem_function_array_error.xdmf", "w") as file:
#             #     file.write_mesh(mesh)
#             #     file.write_function(fem_function_array_error , tmp_cycle)    
    
#             # ##---------------------------------------------------------------------------------------------------
#             # ##---------------------------------------------------------------------------------------------------

#             # Continúa con la parte de resolución de ecuaciones y condiciones de frontera...
#             # Al parecer tambien hay un error en el calculo de mu no es + sino -
#             mu = fem_function_modulo_E_f / (2.0 * (1.0 + 0.3))
#             lame = fem_function_modulo_E_f * 0.3/ ((1.0 + 0.3) * (1.0 - 2.0 * 0.3)) 
#             # revisar el valor de lamnda pues al parecer hay un error  
                                    
#             f = ufl.as_vector((0, 0, -fem_function_densidad * 9810 ))
#             V = functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
#             u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
           
#             def sigma(v):
#                 """Return an expression for the stress σ given a displacement field"""
#                 return 2.0 * mu * ufl.sym(grad(v)) + lame * ufl.tr(ufl.sym(grad(v))) * ufl.Identity(3)

#             def strain(u):
#                 return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
#             a = form(inner(sigma(u), grad(v)) * dx)
#             L = form(inner(f, v) * dx)

#             def invariants_principal(A):
#                 i1 = ufl.tr(A)
#                 i2 = (ufl.tr(A)**2 - ufl.tr(A * A)) / 2
#                 i3 = ufl.det(A)
#                 return i1, i2, i3

#             #----------------------------------------------------------------------------
#             # Asignacion de la condición frontera 08/02/24 - R-ninety
#             #----------------------------------------------------------------------------
#             def boundary_z(x):
#                 return np.logical_and(x[2] >= 36, x[2] <= 37)
            
#             # Ubicar los grados de libertad en las caras con z entre 36 y 37
#             dofs = dolfinx.fem.locate_dofs_geometrical(V, boundary_z)
            
#             # Definir la condición de frontera como desplazamiento 0 en todas las direcciones
#             u_D = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0, 0, 0)))
            
#             # Aplicar la condición de frontera
#             bc = dolfinx.fem.dirichletbc(u_D, dofs, V)
#             #----------------------------------------------------------------------------
            
#             # ## Assemble and solve
#             #
#             # The bilinear form `a` is assembled into a matrix `A`, with
#             # modifications for the Dirichlet boundary conditions. The call
#             # `A.assemble()` completes any parallel communication required to
#             # compute the matrix.


#             A = assemble_matrix(a, bcs=[bc])
#             A.assemble()
            
            

#             # Coordenadas de los puntos donde se aplicarán las cargas puntuales
#             point_138 = np.array([87.7431, 20.6717, 224.6261],dtype=np.float64)
#             point_140 = np.array([85.6372, 33.9296, 226.9851],dtype=np.float64)
#             point_141 = np.array([77.7509, 23.4239, 216.3084],dtype=np.float64)
#             point_143 = np.array([78.43, 17.2494, 209.5734],dtype=np.float64)
#             point_148 = np.array([73.5285, 30.5294, 203.1597],dtype=np.float64)
#             point_52 = np.array([124.2809, 54.4087,180.5598],dtype=np.float64)

#             # Cargas en la dirección Z y X

#             def marker_52(x):
#                 # Asegurarse de que x se compara correctamente con las coordenadas de point_138
#                 close_x = np.isclose(x[0], point_52[0], atol=atol)
#                 close_y = np.isclose(x[1], point_52[1], atol=atol)
#                 close_z = np.isclose(x[2], point_52[2], atol=atol)
#                 return np.logical_and(np.logical_and(close_x, close_y), close_z)

#             def marker_138(x):
#                 # Asegurarse de que x se compara correctamente con las coordenadas de point_138
#                 close_x = np.isclose(x[0], point_138[0], atol=atol)
#                 close_y = np.isclose(x[1], point_138[1], atol=atol)
#                 close_z = np.isclose(x[2], point_138[2], atol=atol)
#                 return np.logical_and(np.logical_and(close_x, close_y), close_z)

#             def marker_140(x):
#                 # Asegurarse de que x se compara correctamente con las coordenadas de point_138
#                 close_x = np.isclose(x[0], point_140[0], atol=atol)
#                 close_y = np.isclose(x[1], point_140[1], atol=atol)
#                 close_z = np.isclose(x[2], point_140[2], atol=atol)
#                 return np.logical_and(np.logical_and(close_x, close_y), close_z)
#             def marker_141(x):
#                 # Asegurarse de que x se compara correctamente con las coordenadas de point_138
#                 close_x = np.isclose(x[0], point_141[0], atol=atol)
#                 close_y = np.isclose(x[1], point_141[1], atol=atol)
#                 close_z = np.isclose(x[2], point_141[2], atol=atol)
#                 return np.logical_and(np.logical_and(close_x, close_y), close_z)
#             def marker_143(x):
#                 # Asegurarse de que x se compara correctamente con las coordenadas de point_138
#                 close_x = np.isclose(x[0], point_143[0], atol=atol)
#                 close_y = np.isclose(x[1], point_143[1], atol=atol)
#                 close_z = np.isclose(x[2], point_143[2], atol=atol)
#                 return np.logical_and(np.logical_and(close_x, close_y), close_z)

#             def marker_148(x):
#                 # Asegurarse de que x se compara correctamente con las coordenadas de point_138
#                 close_x = np.isclose(x[0], point_148[0], atol=atol)
#                 close_y = np.isclose(x[1], point_148[1], atol=atol)
#                 close_z = np.isclose(x[2], point_148[2], atol=atol)
#                 return np.logical_and(np.logical_and(close_x, close_y), close_z)



#             b = assemble_vector(L)

#             # Localiza los grados de libertad en el vértice
#             vertices_52 = locate_dofs_geometrical(V, marker_138)
#             dofs_52_x = (vertices_52[0])*3 + 0  # Grado de libertad en la dirección y
#             dofs_52_z = (vertices_52[0])*3 + 2  # Grado de libertad en la dirección z


#             vertices_138 = locate_dofs_geometrical(V, marker_138)
#             dofs_138_x = (vertices_138[0])*3 + 0  # Grado de libertad en la dirección y
#             dofs_138_z = (vertices_138[0])*3 + 2  # Grado de libertad en la dirección z

#             vertices_140 = locate_dofs_geometrical(V, marker_140)
#             dofs_140_x = (vertices_140[0])*3 + 0  # Grado de libertad en la dirección y
#             dofs_140_z = (vertices_140[0])*3 + 2  # Grado de libertad en la dirección z

#             vertices_141 = locate_dofs_geometrical(V, marker_141)
#             dofs_141_x = (vertices_141[0])*3 + 0  # Grado de libertad en la dirección y
#             dofs_141_z = (vertices_141[0])*3 + 2  # Grado de libertad en la dirección z

#             vertices_143 = locate_dofs_geometrical(V, marker_143)
#             dofs_143_x = (vertices_143[0])*3 + 0  # Grado de libertad en la dirección y
#             dofs_143_z = (vertices_143[0])*3 + 2  # Grado de libertad en la dirección z

#             vertices_148 = locate_dofs_geometrical(V, marker_148)
#             dofs_148_x = (vertices_148[0])*3 + 0  # Grado de libertad en la dirección y
#             dofs_148_z = (vertices_148[0])*3 + 2  # Grado de libertad en la dirección z



#             #print("El valor de los DOFs en X y en Z en la  carga:", dofs_138_x, dofs_138_z)
#             # Magnitudes de las cargas puntuales
#             P_x = 0  # Carga en la dirección y 
#             P_z = -2317   # Carga en la dirección z
#             P_z2= 713       # Carga del abductor en z
#                 # Aplicar las cargas puntuales
#             with b.localForm() as loc_b:
#                 loc_b.setValues(dofs_138_x, P_x)
#                 loc_b.setValues(dofs_138_z, P_z)
#                 loc_b.setValues(dofs_140_x, P_x)
#                 loc_b.setValues(dofs_140_z, P_z) 
#                 loc_b.setValues(dofs_141_x, P_x)
#                 loc_b.setValues(dofs_141_z, P_z) 
#                 loc_b.setValues(dofs_143_x, P_x)
#                 loc_b.setValues(dofs_143_z, P_z) 
#                 loc_b.setValues(dofs_148_x, P_x)
#                 loc_b.setValues(dofs_148_z, P_z)
#                 loc_b.setValues(dofs_52_x, P_x)
#                 loc_b.setValues(dofs_52_z, P_z2)
            
            
#             apply_lifting(b, [a], bcs=[[bc]])
#             b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
#             set_bc(b, [bc])
    
    
#             ns = build_nullspace(V)
#             A.setNearNullSpace(ns)
#             A.setOption(PETSc.Mat.Option.SPD, True)  # type: ignore
    
            
#             # +
#             # Set solver options
#             opts = PETSc.Options()  # type: ignore
#             opts["ksp_type"] = "cg"
#             opts["ksp_rtol"] = 1.0e-8
#             opts["pc_type"] = "gamg"
        
#             # Use Chebyshev smoothing for multigrid
#             opts["mg_levels_ksp_type"] = "chebyshev"
#             opts["mg_levels_pc_type"] = "jacobi"
        
#             # Improve estimate of eigenvalues for Chebyshev smoothing
#             opts["mg_levels_ksp_chebyshev_esteig_steps"] = 10
        
#             # Create PETSc Krylov solver and turn convergence monitoring on
#             solver = PETSc.KSP().create(mesh.comm)  # type: ignore
#             solver.setFromOptions()
    
#             # Set matrix operator
#             solver.setOperators(A)
            
#             # +
#             uh2_in = Function(V)
    
#             solver.solve(b, uh2_in.vector)
    
#             uh2_in.x.scatter_forward()
            
#             # ## Post-processing
#             # +
#             # sigma_dev = sigma(uh_in) - (1 / 3) * ufl.tr(sigma(uh_in)) * ufl.Identity(len(uh_in))
#             # sigma_vm = ufl.sqrt((3 / 2) * inner(sigma_dev, sigma_dev))
#             # W = functionspace(mesh, ("Discontinuous Lagrange", 0))
            
#             # sigma_vm_expr = Expression(sigma_vm, W.element.interpolation_points())
#             # sigma_vm_h = Function(W,name="Von_misses")
#             # sigma_vm_h.interpolate(sigma_vm_expr)

            
#             # ----------------------------------------------------------------------------------

            
           
#             U = 0.5 * ufl.inner(sigma(uh2_in), strain(uh2_in)) * ufl.dx
    
#             # Ensamblar la energía de deformación ----<< 09/02/24 ---<<< OKß
#             energy = fem.assemble_scalar(fem.form(U)) 
#            # print("Energía de deformación total:", energy)
#            # print("Sigma_bar:", sigma_bar_i)
#             print(f"El valor promedio de daño:{promedios_damage[escalar_contador_total_dias]}")
#             print(f"El valor promedio de la densidad:{promedios_rho_cortical_gr_cm_3[escalar_contador_total_dias]}")
#             print(f"El valor promedio de la fracción de volumen:{promedios_BVTV[escalar_contador_total_dias]}")
#             print(f"2.{dday} - El valor promedio del modulo de elasticidad:{promedios_modulo_elasticidad[escalar_contador_total_dias]}")

#     with XDMFFile(mesh.comm, "./20_01_05_24/fase_9/misses_.xdmf", "w") as file:
#         file.write_mesh(mesh)
#         file.write_function(sigma_vm_h , dday)
#     with XDMFFile(mesh.comm, "./20_01_05_24/fase_9/displacements_.xdmf", "w") as file:
#         file.write_mesh(mesh)
#         file.write_function(uh_in, dday)
#     # with XDMFFile(mesh.comm, "./20_01_05_24/fase_7/sigma_.xdmf", "w") as file:
#     #     file.write_mesh(mesh)
#     #     file.write_function(valor_sigma , dday)
#     with XDMFFile(mesh.comm, "./20_01_05_24/fase_9/fem_function_modulo.xdmf", "w") as file:
#         file.write_mesh(mesh)
#         file.write_function(fem_function_modulo_E_f , tmp_cycle)    
#     with XDMFFile(mesh.comm, "./20_01_05_24/fase_9/fem_function_densidad_material.xdmf", "w") as file:
#         file.write_mesh(mesh)
#         file.write_function(fem_function_densidad_material , tmp_cycle)    
#     with XDMFFile(mesh.comm, "./20_01_05_24/fase_9/fem_function_BVTV.xdmf", "w") as file:
#         file.write_mesh(mesh)
#         file.write_function(fem_function_BVTV , tmp_cycle)    
#     with XDMFFile(mesh.comm, "./20_01_05_24/fase_9/fem_function_array_modulo.xdmf", "w") as file:
#         file.write_mesh(mesh)
#         file.write_function(fem_function_modulo_E_f , tmp_cycle)    
#     with XDMFFile(mesh.comm, "./20_01_05_24/fase_9/fem_function_array_vm.xdmf", "w") as file:
#         file.write_mesh(mesh)
#         file.write_function(fem_function_array_vm , tmp_cycle)    
#     with XDMFFile(mesh.comm, "./20_01_05_24/fase_9/fem_function_array_damage.xdmf", "w") as file:
#         file.write_mesh(mesh)
#         file.write_function(fem_function_damage , tmp_cycle)    
#     with XDMFFile(mesh.comm, "./20_01_05_24/fase_9/fem_function_alpha.xdmf", "w") as file:
#          file.write_mesh(mesh)
#          file.write_function(fem_function_alpha , tmp_cycle)    
#     with XDMFFile(mesh.comm, "./20_01_05_24/fase_9/fem_function_array_rdot.xdmf", "w") as file:
#         file.write_mesh(mesh)
#         file.write_function(fem_function_array_rdot , tmp_cycle)    
#     with XDMFFile(mesh.comm, "./20_01_05_24/fase_9/fem_function_array_psi.xdmf", "w") as file:
#         file.write_mesh(mesh)
#         file.write_function(fem_function_array_psi , tmp_cycle)    
#     with XDMFFile(mesh.comm, "./20_01_05_24/fase_9/fem_function_array_error.xdmf", "w") as file:
#         file.write_mesh(mesh)
#         file.write_function(fem_function_array_error , tmp_cycle)    
#     with XDMFFile(mesh.comm, "./20_01_05_24/fase_9/fem_function_densidad_fase_7.xdmf", "w") as file:
#         file.write_mesh(mesh)
#         file.write_function(fem_function_densidad_gr_cm3 , tmp_day)
#     with XDMFFile(mesh.comm, "./20_01_05_24/fase_9/fem_function_BVTV_fase_8.xdmf", "w") as file:
#         file.write_mesh(mesh)
#         file.write_function(fem_function_BVTV , tmp_day)
            
            
#     with XDMFFile(mesh.comm, "./20_01_05_24/fase_9/fem_function_densidad_fase_1.xdmf", "w") as file:
#         file.write_mesh(mesh)
#         file.write_function(fem_function_densidad_gr_cm3 , tmp_day)
#     with XDMFFile(mesh.comm, "./20_01_05_24/fase_9/fem_function_BVTV_fase_1.xdmf", "w") as file:
#         file.write_mesh(mesh)
#         file.write_function(fem_function_BVTV , tmp_day)
#     with XDMFFile(mesh.comm, "./20_01_05_24/fase_9/fem_function_densidad.xdmf", "w") as file:
#         file.write_mesh(mesh)
#         file.write_function(fem_function_densidad_gr_cm3 , tmp_day)
#     with XDMFFile(mesh.comm, "./20_01_05_24/fase_9/fem_function_BVTV.xdmf", "w") as file:
#         file.write_mesh(mesh)
#         file.write_function(fem_function_BVTV , tmp_day)

#     print("---------------------------------------------------------")              
#     print("El valor promedio de la BVTV es:", promedios_BVTV[escalar_contador_total_dias])
#     print(f" Finaliza segundo ciclo etapa -->> La energía: {energy}")
#     print("---------------------------------------------------------")
#     tmp_lenght = escalar_contador_total_dias
#     print(f"02-longitud del CSV-->>>>:{tmp_lenght}")  
    



tmp_lenght = escalar_contador_total_dias
print(f"07-longitud del CSV-->>>>:{tmp_lenght}")
# Crear un DataFrame usando los arrays almacenados
data = pd.DataFrame({
    'Día': np.arange(1, tmp_lenght + 1),
    'Damage Promedio': promedios_damage[:tmp_lenght],
    'Densidad Promedio': promedios_rho_cortical_gr_cm_3[:tmp_lenght],
    'Fracción de Volumen Promedio': promedios_BVTV[:tmp_lenght],
    'Energia de Deformacion': escalar_energy[:tmp_lenght],
    'Modulo de elasticidad promedio': promedios_modulo_elasticidad[:tmp_lenght],
    '% de Cenizas alpha': promedios_alpha[:tmp_lenght],
    'Promedios densida tejido': promedios_rho_tejido[:tmp_lenght]
})


# Exportar el DataFrame a CSV
data.to_csv('./20_01_05_24/resultados_promedio_final.csv', index=False)
print("Archivo CSV guardado con éxito.")


                       

end_time = time.time()

# Hasta este punto hemos podido calcular los valores de los módulos de elasticidad

     # Calcular la duración
duration = end_time - start_time

print(f"El tiempo empleado en el cálculo fue de {duration} segundos.")
