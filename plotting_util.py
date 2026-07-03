'''
Shaun P.

Plotting utility script for Work Experience Week at Diamond.
'''
# %% import some useful packages
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
import numpy as np

# %% apply settings
plt.rcParams['font.size'] = 12 # font size of labels and misc. items on plots.
plt.rcParams['text.usetex'] = True # use LaTeX (a special typesetting font used in academic papers).
labelsize = 14

# %% Function definitons
# Quadrupole matrix
def M_Quadrupole(k: float, L: float) -> np.ndarray:
    '''
    Accepts:
        * k: The quadrupole strength.
        * L: The length of the 'thin' quadrupole.
    Returns:
        * A 2D array of shape (2, 2) representing the transfer matrix for a thin quadrupole.
    '''
    return np.array([
        [1, 0],
        [k * L, 1],
    ])

# Drift matrix
def M_Drift(L: float) -> np.ndarray:
    '''
    Accepts:
        * L: The length of the drift space.
    Returns:
        * A 2D array of shape (2, 2) representing the transfer matrix for a drift space.
    '''
    return np.array([
        [1, L],
        [0, 1],
    ])

# Dipole Matrix
def M_Dipole(theta: float, R: float) -> np.ndarray:
    pass

# Calculate the beta function at different s coordinates around a lattice.
def calculate_beta(initial_beta: list, n_FODO: int, k_abs: float, quad_length: float, drift_length: float, n_subdrifts: int = 1) -> np.ndarray:
    '''
    Accepts:
        * `initial_beta`: length 2 array of initial beta functions in the x and y planes, respectively.
        * Number of FODO cells
    Returns:
        * Beta function value at s = L in _metres_
    '''
    if not initial_beta[0] > 0 or not initial_beta[1] > 0:
        return ValueError('Initial beta(s) must be positive.')

    # betas = np.zeros(5 * n_FODO + 1)
    betas = np.zeros(((3 + 2 * n_subdrifts) * n_FODO + 1, 2))
    betas[0] = initial_beta
    elems = []
    s = [0]
    subdrift_length = drift_length / n_subdrifts
    for n in range(n_FODO):
        # subdivide the drifts into smaller connected drifts to get better approximations of the true beta evolution.
        subdrifts = [M_Drift(subdrift_length) for n in range(n_subdrifts)]
        sub_s = [subdrift_length for n in range(n_subdrifts)]
        # focusing quads on the ends should have have the length as the central defocusing quad.
        # .extend() is a method that adds extra items to the end of a list in Python.
        elems.extend([M_Quadrupole(-k_abs, quad_length / 2), *subdrifts, M_Quadrupole(k_abs, quad_length), *subdrifts, M_Quadrupole(-k_abs, quad_length / 2)])
        s.extend([quad_length / 2, *sub_s, quad_length, *sub_s, quad_length / 2])

    # there is an additional Twiss vector with three elements (each related to the trace-space ellipse).
    twiss_state = np.array([
        [initial_beta[0]], # beta
        [0], # alpha
        [1 / initial_beta[0]], # gamma
    ])

    # Twiss parameters describe the properties of the beam as a whole.
    # propagate the Twiss vector forwards at each element / sub-element.
    for it, M in enumerate(elems):
        # it is possible to construct a 3 x 3 Twiss matrix using elements of a transfer matrix, which you are familiar with.
        twiss_state = np.array([
            [M[0, 0] ** 2, -2 * M[0, 0] * M[0, 1], M[0, 1] ** 2],
            [-M[0, 0] * M[1, 0], M[0, 1] * M[1, 0] + M[1, 1] * M[0, 0], -M[0, 1] * M[1, 1]],
            [M[1, 0] ** 2, -2 * M[1, 1] * M[1, 0], M[1, 1] ** 2],
        ]) @ twiss_state # @ is the symbol for matrix multiplication.
        betas[it + 1][0] = twiss_state[0, 0] # store this new beta value (first element in the Twiss vector) in the betas array.

    # repeat the process for the vertical plane and swap the signs on the quads.
    elems = []
    for n in range(n_FODO):
        elems.extend([M_Quadrupole(k_abs, quad_length / 2), *subdrifts, M_Quadrupole(-k_abs, quad_length), *subdrifts, M_Quadrupole(k_abs, quad_length / 2)])
    
    twiss_state = np.array([
        [initial_beta[1]], # beta
        [0], # alpha
        [1 / initial_beta[1]], # gamma
    ])

    for it, M in enumerate(elems):
        twiss_state = np.array([
            [M[0, 0] ** 2, -2 * M[0, 0] * M[0, 1], M[0, 1] ** 2],
            [-M[0, 0] * M[1, 0], M[0, 1] * M[1, 0] + M[1, 1] * M[0, 0], -M[0, 1] * M[1, 1]],
            [M[1, 0] ** 2, -2 * M[1, 1] * M[1, 0], M[1, 1] ** 2],
        ]) @ twiss_state
        betas[it + 1][1] = twiss_state[0, 0]

    # np.cumsum() stores the sum up to the current element, for each element in an array.
    return betas, np.cumsum(s)

def construct_lattice_full(bend_angle_in_deg: float, radius_in_metres: float, initial_beta: float, k_abs: float, quad_length: float, drift_length: float, n_straight_FODOs: int, n_subdrifts = 1, QF_color: str = '#7A6EE6', QD_color = '#60CF51', QF_edge_color: str = 'blue', QD_edge_color: str = 'green') -> list[Axes]:
    # The multi-line comment below is a docstring.
    '''
    Accepts:
        * Bend angle per FODO cell in _degrees_
        * Bend radius in _metres_
        * Initial beta function value in _metres_
    Returns:
        None: Plot of a periodic FODO lattice along with beta function evolution.
    '''
    if 360 % bend_angle_in_deg != 0:
        return ValueError("360 degrees must be cleanly divisible by the bend angle.")
    bend_angle_in_radians = bend_angle_in_deg * np.pi / 180
    fig = plt.figure(figsize = (16, 4.75))

    # create a gridspec to hold subplots (1 row, 2 columns)
    gs = GridSpec(1, 2, figure = fig, width_ratios = [1, 2])
    # add some horizontal space between the subplots
    gs.update(wspace = .2)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])

    # add axis labels - x
    ax00.set_xlabel(r'$x_{lab}~\mathrm{[m]}$', fontsize = labelsize)
    ax00.set_ylabel(r'$z_{lab}~\mathrm{[m]}$', fontsize = labelsize)
    # add axis labels - y
    ax01.set_xlabel(r'$s~\mathrm{[m]}$', fontsize = labelsize)
    ax01.set_ylabel(r'$\beta_x / \beta_y~\mathrm{[m]}$', fontsize = labelsize)

    # plot the FODO elements
    n_FODO = int(360 / bend_angle_in_deg)
    print(f'{n_FODO} cells in this lattice.')

    straight_length = n_straight_FODOs * drift_length
    yrng = radius_in_metres + straight_length

    half_width = 0.035 * yrng
    half_height = 0.115 * yrng

    # Defocusing quadrupole
    QD = [
        (Path.MOVETO, (-half_width, half_height)),
        (Path.LINETO, (half_width, half_height)),
        (
            Path.CURVE3,
            (-.1, 0),
        ),
        (
            Path.CURVE3,
            (half_width, -half_height),
        ),
        (Path.LINETO, (-half_width, -half_height)),
        (
            Path.CURVE3,
            (.1, 0),
        ),
        (
            Path.CURVE3,
            (-half_width, half_height),
        ),
        (Path.CLOSEPOLY, (0, 0)),  # Close the loop
    ]
    # Focusing quadrupole
    QF = [
        (Path.MOVETO, (-.01 * half_width, half_height)),
        (Path.LINETO, (.01 * half_width, half_height)),
        (
            Path.CURVE3,
            (1.5 * half_width, 0),
        ),
        (
            Path.CURVE3,
            (.01 * half_width, -half_height),
        ),
        (Path.LINETO, (-.01 * half_width, -half_height)),
        (
            Path.CURVE3,
            (-1.5 * half_width, 0),
        ),
        (
            Path.CURVE3,
                (-.01 * half_width, half_height),
            ),
        (Path.CLOSEPOLY, (0, 0)),  # Close the loop
    ]

    # create a shape for each quad.
    def get_Q(Q, include_label = False):
        label = ''
        if include_label:
            label = 'QF' if Q == QF else 'QD'
        codes, verts = zip(*Q)
        return patches.PathPatch(
            Path(verts, codes),
            facecolor = "#7A6EE6" if Q == QF else "#60CF51",
            edgecolor = 'blue' if Q == QF else 'green',
            lw = 1,
            label = label,
        )

    pad_factor = 1.2
    ax00.set_xlim(-pad_factor * yrng, pad_factor * yrng)
    ax00.set_ylim(-pad_factor * yrng, pad_factor * yrng)

    QF_plotted = False
    QD_plotted = False

    top_arc = patches.Arc(
        (0, 0),
        2 * radius_in_metres,
        2 * radius_in_metres,
        angle = 0,
        theta1 = 0, theta2 = 180,
        edgecolor = 'black',
        lw = 1,
    )
    top_arc.set_transform(Affine2D().translate(0, straight_length) + ax00.transData)
    ax00.add_patch(top_arc)

    bottom_arc = patches.Arc(
        (0, 0),
        2 * radius_in_metres,
        2 * radius_in_metres,
        angle = 0,
        theta1 = 0, theta2 = 180,
        edgecolor = 'black',
        lw = 1,
    )
    bottom_arc.set_transform(Affine2D().rotate(np.pi).translate(0, -straight_length) + ax00.transData)
    ax00.add_patch(bottom_arc)

    # left connecting line
    ax00.plot([-radius_in_metres, -radius_in_metres], [-straight_length, straight_length], color = 'black', lw = 1, zorder = -1)
    ax00.plot([radius_in_metres, radius_in_metres], [-straight_length, straight_length], color = 'black', lw = 1, zorder = -1)

    for _ in range(int(n_FODO / 2)):
        # bottom
        Q = get_Q(QF, include_label = not QF_plotted)
        Q.set_transform(Affine2D().translate(0, radius_in_metres).rotate(_ * bend_angle_in_radians).rotate(np.pi / 2).translate(0, -straight_length) + ax00.transData)
        ax00.add_patch(Q)

        Q = get_Q(QD, include_label = not QD_plotted)
        Q.set_transform(Affine2D().translate(0, radius_in_metres).rotate(_ * bend_angle_in_radians + bend_angle_in_radians / 2).rotate(np.pi / 2).translate(0, -straight_length) + ax00.transData)
        ax00.add_patch(Q)

        # top
        Q = get_Q(QF)
        Q.set_transform(Affine2D().translate(0, radius_in_metres).rotate(_ * bend_angle_in_radians).rotate(-np.pi / 2).translate(0, straight_length) + ax00.transData)
        ax00.add_patch(Q)

        Q = get_Q(QD)
        Q.set_transform(Affine2D().translate(0, radius_in_metres).rotate(_ * bend_angle_in_radians + bend_angle_in_radians / 2).rotate(-np.pi / 2).translate(0, straight_length) + ax00.transData)
        ax00.add_patch(Q)

        if not QF_plotted:
            QF_plotted = True
            QD_plotted = True

    # plot the quadrupoles in the straight sections
    for _ in range(n_straight_FODOs):
        FODO_offset = _ * (2 * drift_length)
        # left from top to bottom
        Q = get_Q(QF)
        Q.set_transform(Affine2D().rotate(np.pi / 2).translate(-radius_in_metres, straight_length - FODO_offset) + ax00.transData)
        ax00.add_patch(Q)

        Q = get_Q(QD)
        Q.set_transform(Affine2D().rotate(np.pi / 2).translate(-radius_in_metres, straight_length - FODO_offset - drift_length) + ax00.transData)
        ax00.add_patch(Q)

        # right from top to bottom
        Q = get_Q(QF)
        Q.set_transform(Affine2D().rotate(np.pi / 2).translate(radius_in_metres, -straight_length + FODO_offset) + ax00.transData)
        ax00.add_patch(Q)

        Q = get_Q(QD)
        Q.set_transform(Affine2D().rotate(np.pi / 2).translate(radius_in_metres, -straight_length + FODO_offset + drift_length) + ax00.transData)
        ax00.add_patch(Q)

    ax00.legend(loc = 'center', fontsize = 11)

    betas, s = calculate_beta(initial_beta, n_FODO + n_straight_FODOs, k_abs, quad_length, drift_length, n_subdrifts)

    # plot the beta function
    ax01.plot(s, betas[:, 0], color = 'tab:blue', label = r'$\beta_x$')
    ax01.plot(s, betas[:, 1], color = 'tab:red', label = r'$\beta_y$')
    ax01.set_xlim(0, s[-1])
    ax01.set_ylim(0, None)
    ax01.legend(ncols = 2, fontsize = 12)

    # save the figure
    plt.savefig('FODO_ring.png', dpi = 400, bbox_inches = 'tight')

    return [ax00, ax01]

def plot_lattice(bend_angle_in_deg: float, radius_in_metres: float, n_straight_FODOs: int, QF_color: str = '#7A6EE6', QD_color = '#60CF51', QF_edge_color: str = 'blue', QD_edge_color: str = 'green') -> Axes:
    if 360 % bend_angle_in_deg != 0:
        return ValueError("360 degrees must be cleanly divisible by the bend angle.")
    bend_angle_in_radians = bend_angle_in_deg * np.pi / 180
    fig = plt.figure(figsize = (5, 4.75))

    # create a gridspec to hold subplots (1 row, 2 columns)
    gs = GridSpec(1, 1, figure = fig)
    # add some horizontal space between the subplots
    ax00 = fig.add_subplot(gs[0, 0])

    # add axis labels
    ax00.set_xlabel(r'$x_{lab}~\mathrm{[m]}$', fontsize = labelsize)
    ax00.set_ylabel(r'$z_{lab}~\mathrm{[m]}$', fontsize = labelsize)

    # plot the FODO elements
    n_FODO = int(360 / bend_angle_in_deg)
    print(f'{n_FODO} cells in this lattice.')

    straight_length = n_straight_FODOs * drift_length
    yrng = radius_in_metres + straight_length

    half_width = 0.035 * yrng
    half_height = 0.115 * yrng
    # Defocusing quadrupole
    QD = [
        (Path.MOVETO, (-half_width, half_height)),
        (Path.LINETO, (half_width, half_height)),
        (
            Path.CURVE3,
            (-.1, 0),
        ),
        (
            Path.CURVE3,
            (half_width, -half_height),
        ),
        (Path.LINETO, (-half_width, -half_height)),
        (
            Path.CURVE3,
            (.1, 0),
        ),
        (
            Path.CURVE3,
            (-half_width, half_height),
        ),
        (Path.CLOSEPOLY, (0, 0)),  # Close the loop
    ]
    # Focusing quadrupole
    QF = [
        (Path.MOVETO, (-.01 * half_width, half_height)),
        (Path.LINETO, (.01 * half_width, half_height)),
        (
            Path.CURVE3,
            (1.5 * half_width, 0),
        ),
        (
            Path.CURVE3,
            (.01 * half_width, -half_height),
        ),
        (Path.LINETO, (-.01 * half_width, -half_height)),
        (
            Path.CURVE3,
            (-1.5 * half_width, 0),
        ),
        (
            Path.CURVE3,
                (-.01 * half_width, half_height),
            ),
        (Path.CLOSEPOLY, (0, 0)),  # Close the loop
    ]

    # create a shape for each quad.
    def get_Q(Q, include_label = False):
        label = ''
        if include_label:
            label = 'QF' if Q == QF else 'QD'
        codes, verts = zip(*Q)
        return patches.PathPatch(
            Path(verts, codes),
            facecolor = QF_color if Q == QF else QD_color,
            edgecolor = QF_edge_color if Q == QF else QD_edge_color,
            lw = 1,
            label = label,
        )

    pad_factor = 1.2
    ax00.set_xlim(-pad_factor * yrng, pad_factor * yrng)
    ax00.set_ylim(-pad_factor * yrng, pad_factor * yrng)

    QF_plotted = False
    QD_plotted = False

    top_arc = patches.Arc(
        (0, 0),
        2 * radius_in_metres,
        2 * radius_in_metres,
        angle = 0,
        theta1 = 0, theta2 = 180,
        edgecolor = 'black',
        lw = 1,
    )
    top_arc.set_transform(Affine2D().translate(0, straight_length) + ax00.transData)
    ax00.add_patch(top_arc)

    bottom_arc = patches.Arc(
        (0, 0),
        2 * radius_in_metres,
        2 * radius_in_metres,
        angle = 0,
        theta1 = 0, theta2 = 180,
        edgecolor = 'black',
        lw = 1,
    )
    bottom_arc.set_transform(Affine2D().rotate(np.pi).translate(0, -straight_length) + ax00.transData)
    ax00.add_patch(bottom_arc)

    # left connecting line
    ax00.plot([-radius_in_metres, -radius_in_metres], [-straight_length, straight_length], color = 'black', lw = 1, zorder = -1)
    ax00.plot([radius_in_metres, radius_in_metres], [-straight_length, straight_length], color = 'black', lw = 1, zorder = -1)

    for _ in range(int(n_FODO / 2)):
        # bottom
        Q = get_Q(QF, include_label = not QF_plotted)
        Q.set_transform(Affine2D().translate(0, radius_in_metres).rotate(_ * bend_angle_in_radians).rotate(np.pi / 2).translate(0, -straight_length) + ax00.transData)
        ax00.add_patch(Q)

        Q = get_Q(QD, include_label = not QD_plotted)
        Q.set_transform(Affine2D().translate(0, radius_in_metres).rotate(_ * bend_angle_in_radians + bend_angle_in_radians / 2).rotate(np.pi / 2).translate(0, -straight_length) + ax00.transData)
        ax00.add_patch(Q)

        # top
        Q = get_Q(QF)
        Q.set_transform(Affine2D().translate(0, radius_in_metres).rotate(_ * bend_angle_in_radians).rotate(-np.pi / 2).translate(0, straight_length) + ax00.transData)
        ax00.add_patch(Q)

        Q = get_Q(QD)
        Q.set_transform(Affine2D().translate(0, radius_in_metres).rotate(_ * bend_angle_in_radians + bend_angle_in_radians / 2).rotate(-np.pi / 2).translate(0, straight_length) + ax00.transData)
        ax00.add_patch(Q)

        if not QF_plotted:
            QF_plotted = True
            QD_plotted = True

    # plot the quadrupoles in the straight sections
    for _ in range(n_straight_FODOs):
        FODO_offset = _ * (2 * drift_length)
        # left from top to bottom
        Q = get_Q(QF)
        Q.set_transform(Affine2D().rotate(np.pi / 2).translate(-radius_in_metres, straight_length - FODO_offset) + ax00.transData)
        ax00.add_patch(Q)

        Q = get_Q(QD)
        Q.set_transform(Affine2D().rotate(np.pi / 2).translate(-radius_in_metres, straight_length - FODO_offset - drift_length) + ax00.transData)
        ax00.add_patch(Q)

        # right from top to bottom
        Q = get_Q(QF)
        Q.set_transform(Affine2D().rotate(np.pi / 2).translate(radius_in_metres, -straight_length + FODO_offset) + ax00.transData)
        ax00.add_patch(Q)

        Q = get_Q(QD)
        Q.set_transform(Affine2D().rotate(np.pi / 2).translate(radius_in_metres, -straight_length + FODO_offset + drift_length) + ax00.transData)
        ax00.add_patch(Q)
    
    ax00.legend(loc = 'center', fontsize = 10)
    fig.savefig('test.png', dpi = 400, bbox_inches = 'tight')

# %% Example of generating a Diamond Booster lattice figure
drift_length = 2.16
n_FODO = 18
n_straight_FODOs = 1
angle = 360 / n_FODO
radius = (2 * drift_length) * n_FODO / (2 * np.pi)
plot_lattice(angle, radius, n_straight_FODOs)

# %% Example of generating a Diamond Booster lattice figure with accompanying beta plot
drift_length = 2.16
quad_length = .2
n_FODO = 18
n_straight_FODOs = 3
angle = 360 / n_FODO
radius = (2 * drift_length) * n_FODO / (2 * np.pi)
ax00, ax01 = construct_lattice_full(angle, radius, [17.071, 3], 1.4142135, quad_length, drift_length, n_straight_FODOs, 30)