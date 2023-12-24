#! /usr/bin/env python3
"""
Produce floats and data for OGLS manuscript
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pylab import *
from scipy.stats import chi2, kstest, norm
from matplotlib import ticker
from matplotlib.markers import MarkerStyle
import ogls_snapshot as ogls
from D47calib_snapshot import *
from pathlib import Path
import shutil

pfactor = chi2.ppf(0.95, 1)**.5

style.use('mydefault.mplstyle')
yello = (1,.8,.2)
hdiamond = MarkerStyle("d")
hdiamond._transform.rotate_deg(90)

calibs = {
	anderson_2021_lsce: dict(
		marker = 'o',
		ms = 4,
		zorder = 200,
		label = '$\\mathcal{[A21·LSCE]}$ (Devils Hole & Laghetto Basso)',
		ls = (0, (6,2)),
		subplot = [4, 4, 15],
		ecolor = (.75,0,.75),
		),
	anderson_2021_mit: dict(
		marker = 'd',
		ms = 4,
		moz = 'A',
		label = '$\\mathcal{[A21·MIT]}$ (diverse carbonates)',
		ls = (0, (6,2)),
		subplot = [4, 4, 1],
		ecolor = (1,.5,0),
		),
	breitenbach_2018: dict(
		marker = 'v',
		ms = 4,
		moz = 'B',
		label = '$\\mathcal{[B18]}$ (cave pearls)',
		ls = (0, (6,2,2,2)),
		subplot = [4, 4, 7],
		ecolor = (.5,.5,.5),
		),
	fiebig_2021: dict(
		marker = 's',
		ms = 3.6,
		moz = 'F',
		label = '$\\mathcal{[F21]}$ (natural & synthetic calcites)',
		ls = (0, (6,2,2,2)),
		subplot = [4, 4, 5],
		ecolor = (1,0,0),
		),
	huyghe_2022: dict(
		marker = hdiamond,
		ms = 4,
		moz = 'H',
		label = '$\\mathcal{[H22]}$ (marine calcitic bivalves)',
		ls = (0, (2,1)),
		subplot = [4, 4, 11],
		ecolor = (0,.75,0),
		),
	jautzy_2020: dict(
		marker = 'D',
		ms = 3.2,
		moz = 'J',
		label = '$\\mathcal{[J20]}$ (synthetic calcites)',
		ls = (0, (6,2,2,2,2,2)),
		subplot = [4, 4, 9],
		ecolor = (0,.8,1),
		),
	peral_2018: dict(
		marker = '^',
		ms = 4,
		moz = 'P',
		label = '$\\mathcal{[P18]}$ (planktic  foraminifera)',
		ls = (0, (6,2,2,2,2,2)),
		subplot = [4, 4, 3],
		ecolor = (0,.5,1),
		),
	devils_laghetto_2023: dict(
		marker = 'o',
		ms = 4,
		moz = 'D',
		label = '$\\mathcal{[DL23]}$ (Devils Hole & Laghetto Basso)',
		ls = (0, (1,4,1,1)),
		subplot = [4, 4, 15],
		),
	}

def fig_pearson_york():
	from yorkregression import YorkReg
	from csv import DictReader

	with open('pearson_york_data.csv') as fid:
		pearson_york_data = [{k: float(_[k]) for k in _} for _ in DictReader(fid)]
	
	for _ in pearson_york_data:
		_['sX'] = _['wX']**-0.5
		_['sY'] = _['wY']**-0.5

	X, Y, sX, sY = zip(*[(_['X'], _['Y'], _['sX'], _['sY']) for _ in pearson_york_data])

	a_y, b_y, CM_y = YorkReg(X, Y, sX, sY, cvg_limit=1e-9, max_iter = 1e3)

	A = ogls.Polynomial(X = X, Y = Y, sX = sX, sY = sY, degrees = [0,1])
	A.regress()


	fig = figure(figsize = (3.5,4.5))
	subplots_adjust(.13,.12,.96,.96)

	A.plot_error_bars(ecolor = 'k', alpha = 1)
	x1, x2 = axis()[:2]
	xi = np.linspace(x1-1, x2+1, 200)

	A.plot_bff_ci(xi, color = yello, alpha = .25, zorder = -200)
	line_pr, = A.plot_bff(xi, color = yello, lw = 4, solid_capstyle = 'butt', zorder = -100)

	yi = a_y * xi + b_y
	eyi = chi2.ppf(0.95, 1)**.5 * (CM_y[0,0] * xi**2 + 2*CM_y[0,1]*xi + CM_y[1,1])**.5
	plot(xi, yi-eyi, color = 'k', lw = 1, dashes = (3,3), solid_capstyle = 'butt')
	plot(xi, yi+eyi, color = 'k', lw = 1, dashes = (3,3), solid_capstyle = 'butt')

	line_y, = plot(xi, yi, 'k-', lw = 1, solid_capstyle = 'butt')

	A.plot_data(marker = 'o', mec = 'k', ms = 5, zorder = 300)
	axis([xi[0], xi[-1], -1.3, None])

	text(0.49, -0.1, '$\\mathit{x}$', va = 'center', ha = 'center', transform = gca().transAxes, size = 11)
	text(-.13, 0.43, '$\\mathit{y}$', va = 'center', ha = 'center', transform = gca().transAxes, size = 11)

	gca().xaxis.set_major_locator(ticker.MultipleLocator(3))
	gca().yaxis.set_major_locator(ticker.MultipleLocator(2))

	legend(
		[(
			plot([],[],'k-', solid_capstyle = 'butt', lw = 10, dashes = (3/10,3/10))[0],
			plot([],[],'w-', lw = 8)[0],
			line_y,
		), (
			plot([],[],'-', color = yello, alpha = .25, lw = 9, solid_capstyle = 'butt')[0],
			line_pr,
		)],
		[
			f'York: slope = {a_y:.3f} ± {CM_y[0,0]**.5:.3f}\nzero-intercept = {b_y:.3f} ± {CM_y[1,1]**.5:.3f}'.replace('-0.', '–$\\,$0.'),
			f'OGLS: slope = {A.bfp["a1"]:.3f} ± {A.bfp_CM[1,1]**.5:.3f}\nzero-intercept = {A.bfp["a0"]:.3f} ± {A.bfp_CM[0,0]**.5:.3f}'.replace('-0.', '–$\\,$0.'),
		],
		loc = 'lower left',
		fontsize = 9,
		bbox_to_anchor = (0,0),
		handlelength = 4,
		labelspacing = 1.5,
		frameon = False,
		ncol = 1,
		)

	fig.savefig('output/fig-04.pdf')
	close(fig)
	

def fig_example_sYX():

	X = [-1, 0, 1, 2]
	Y = [0, 1, 0, -1.5]
	sX = 0.2
	sY = 0.3

	fig = figure(figsize = (3,4))
	subplots_adjust(.08,.06,.95,.97,.2,.2)

	for sp,r,l in [(211, 0.9, 'A'), (212, -0.9, 'B')]:
		subplot(sp)
		sYX = np.eye(len(X)) * r * sX * sY
		P = ogls.Polynomial(X = X, Y = Y, sX = sX, sY = sY, sYX = sYX, degrees = [0,1,2])
		P.regress(params = dict(a0 = 1, a1 = 0, a2 = -1))
		P.plot_error_ellipses(ec = 'k', alpha = .5, zorder = 100)
		P.plot_data(marker = 's', mec = 'k', mfc = 'w', zorder = 200)
		
		x1, x2, y1, y2 = axis()
		
		xi = np.linspace(1.5*x1, 1.5*x2, 101)
		
		P.plot_bff_ci(xi, color = yello, alpha = .25)
		P.plot_bff_ci(xi, fill = False, lw = 1, color = 'k', ls = (0, (6,2,2,2)))
		P.plot_bff(xi, color = yello, lw = 3)

		axis([1.2*x1, 1.2*x2, 1.4*y1, 1.1*y2])

# 		text(.96, .95, l, va = 'top', ha = 'right', weight = 'bold', size = 12, alpha = .5, transform = gca().transAxes)
		text(0.4, .1, f'$\\mathit{{ρ_{{xy}}}} = {r:+}$', va = 'bottom', ha = 'center', weight = 'bold', size = 12, alpha = .5, transform = gca().transAxes)

		xticks([])
		yticks([])
		xlabel('$\\mathit{x}$')
		ylabel('$\\mathit{y}$', rotation = 0, labelpad = 10)

	savefig('output/fig-05.pdf')


def fig_example_sY():
	X = [-1, 0, 1]
	Y = [1, 0, -1]
	sX = [0.1, 0.1, 0.1]
	sy = [0.5, 1.0, 0.5]

	fig = figure(figsize = (3,6))
	subplots_adjust(.08,.06,.95,.97,.2,.2)

	for sp,r,l in [(311, 0.9, 'A'), (312, 0, 'B'), (313, -0.9, 'C')]:
		subplot(sp)
		sY = np.array([[sy[0]**2, 0, r*sy[0]*sy[2]], [0, sy[1]**2, 0], [r*sy[0]*sy[2], 0, sy[2]**2]])
		P = ogls.Polynomial(X = X, Y = Y, sX = sX, sY = sY, degrees = [0,1])
		P.regress()
		P.plot_bff_ci(xi = np.linspace(-2,2,101), color = yello, alpha = .25)
		P.plot_bff_ci(xi = np.linspace(-2,2,101), fill = False, lw = 1, color = 'k', ls = (0, (6,2,2,2)))
		P.plot_bff(xi = np.linspace(-2,2,101), color = yello, lw = 3)
		P.plot_error_ellipses(ec = 'k', alpha = .5, zorder = 100)
		P.plot_data(mec = 'k', zorder = 200)
		for k in range(P.N):
			text(P.X[k], P.Y[k]+0.6, f'{k+1}', va = 'center', ha = 'center', size = 8)
		text(.96, .95, l, va = 'top', ha = 'right', weight = 'bold', size = 12, alpha = .5, transform = gca().transAxes)
		if r > 0:
			s = '\nstrong positive correlation'
		elif r < 0:
			s = '\nstrong negative correlation'
		else:
			s = '\nno correlation'
		text(.04, .05, f'$\\mathit{{ρ_{{y_1y_3}}}}$ = {f"{r:+.1f}" if r else f"{r:.0f}"}{s}'.replace('-', '–'), va = 'bottom', ha = 'left', size = 10, alpha = 1, transform = gca().transAxes)
		xticks([])
		yticks([])
		xlabel('$\\mathit{x}$')
		ylabel('$\\mathit{y}$', rotation = 0, labelpad = 10)
		axis([-2, 2, -4, 4])

	savefig('output/fig-06.pdf')


def fig_convergence_issue():
	f = lambda X: X**3 - 12*X
	X = array([-2,0,2,3,4])
	Y = f(X)
	sX = X*0 + .2
	sY = Y*0 + 2

	Y[-3] += 10
	sY[-3] = 1
	sX[-3] = 1.1

	fig = figure(figsize = (3,3))
	subplots_adjust(.08,.08,.95,.95)

	P = ogls.Polynomial(X = X, Y = Y, sX = sX, sY = sY, degrees = [0,1,2,3])
	P.regress()
	xi = linspace(X[0]-1,X[-1]+1,101)
# 	P.plot_bff_ci(xi = xi, color = yello, alpha = .25)
# 	P.plot_bff_ci(xi = xi, fill = False, lw = 1, color = 'k', ls = (0, (6,2,2,2)))
	plot(xi, f(xi), 'k-', dashes = (8,2,3,2), alpha = .35, lw = 1.5, label = 'true function')
	P.plot_bff(xi = xi, color = yello, lw = 3, label = 'OGLS regression', zorder = -1)
# 	P.plot_error_ellipses(ec = 'k', alpha = .5, zorder = 100)
	P.plot_error_bars(ecolor = 'k', alpha = 1, zorder = 100, capsize = 2)
	P.plot_data(mec = 'k', zorder = 200)
	
	for k, (x,y) in enumerate(zip(X, Y)):
		text(x, y,
			f'\n {k+1}' if x>2 else (f'{k+1} \n' if x==2 else f' {k+1}\n'),
			ha = 'right' if x == 2 else 'left',
			va = 'top' if x>2 else 'bottom',
			linespacing = 0.05,
			alpha = .35,
			)

	xticks([])
	yticks([])
	xlabel('$\\mathit{x}$')
	ylabel('$\\mathit{y}$', rotation = 0, labelpad = 10)
	legend(frameon = False, handlelength = 3.5)
	axis([X[0]-0.9, X[-1]+0.9, Y.min()-10, Y.max()+20])

	savefig('output/fig-07.pdf')


def reorder_matrix_for_compact_blocks(M):
	index_in = list(range(M.shape[0]))
	index_out = []
	index_temp = []
	while len(index_in):
		index_out.append(index_in.pop(0))
		for i in index_in:
			if M[index_out[-1], i]:
				index_out.append(i)
			else:
				index_temp.append(i)
		index_in = index_temp[:]
		index_temp = []
	return array([[M[i,j] for i in index_out] for j in index_out]), index_out
		

def D47correl(calib, filename, vmin = -1, vmax = 1):

	V = calib.sD47
	_ = diag(diag(V)**-.5)
	C = _ @ V @ _
	C, i = reorder_matrix_for_compact_blocks(C)
# 	print(sorted(C.flatten()))

	import matplotlib as mpl
	from matplotlib.colors import ListedColormap, PowerNorm
# 	cm = mpl.colormaps['Spectral'].resampled(256)
# 	newcolors = cm(linspace(0, 1, 256))
# 	newcolors[0, -1] = 0

	negativecolor = [51/255, 51/255, 153/255]
	positivecolor = yello
	overpositivecolor = [255/255, 153/255, 0/255]
		
	newcmp = (
		[[*negativecolor, a**.7] for a in linspace(1,0,int(abs(vmin/vmax)*512))]
		+ [[*positivecolor, a**.7] for a in linspace(0,1,256)]
		+ [[*positivecolor, 1] for a in linspace(0,1,256)]
		)

	for _ in range(256):
		newcmp[_-256] = array(newcmp[_-256]) * (1-_/255) + _/255 * array([*overpositivecolor, 1])

	newcmp = ListedColormap(newcmp)

	newcmp.set_over([.667]*3)
# 	newcmp.set_under([0.4]*3)

	fig = figure(figsize = (4,4))
	subplots_adjust(.15,.1,.9,.85)
	ax = subplot(111)

	_im_ = imshow(C, interpolation = 'nearest', cmap = newcmp, vmin = vmin, vmax = vmax)

	colorbar(_im_, fraction = 0.04, pad = 0.1, location = 'bottom',
# 		format = lambda x, _: f'{x:+.1f}',
		)

	text(.5, -.05, 'Correlation in Δ$_{47}$ measurements', transform = ax.transAxes, va = 'top', ha = 'center', size = 10)

	xlabel('Sample index', labelpad = 11)
	ax.xaxis.set_label_position('top')
	ylabel('Sample index')
	ax.tick_params(top = True, labeltop = True, bottom = False, labelbottom = False)
	gca().xaxis.set_major_locator(ticker.MultipleLocator(10))
	gca().yaxis.set_major_locator(ticker.MultipleLocator(10))
	grid(False)
	fig.savefig(f'output/{filename}.pdf', dpi = 1000)
	close(fig)	


def Tcorrel(calib, filename, vmin = 0, vmax = 1, reorder = False):

	V = calib.sT
	_ = diag(diag(V)**-.5)
	C = _ @ V @ _ + eye(V.shape[0])
	if reorder:
		C, i = reorder_matrix_for_compact_blocks(C)

	import matplotlib as mpl
	from matplotlib.colors import ListedColormap, PowerNorm

	positivecolor = yello
	overpositivecolor = [255/255, 153/255, 0/255]
		
	newcmp = (
		[]
		+ [[*positivecolor, a**.7] for a in linspace(0,1,256)]
		+ [[*positivecolor, 1] for a in linspace(0,1,256)]
		)

	for _ in range(256):
		newcmp[_-256] = array(newcmp[_-256]) * (1-_/255) + _/255 * array([*overpositivecolor, 1])

	newcmp = ListedColormap(newcmp)

	newcmp.set_over([.667]*3)

	fig = figure(figsize = (4,4))
	subplots_adjust(.15,.1,.9,.85)
	ax = subplot(111)

	_im_ = imshow(C, interpolation = 'nearest', cmap = newcmp, vmin = vmin, vmax = vmax)

	colorbar(_im_, fraction = 0.04, pad = 0.1, location = 'bottom',
# 		format = lambda x, _: f'{x:+.1f}',
		)

	text(.5, -.05, 'Correlation in T estimates', transform = ax.transAxes, va = 'top', ha = 'center', size = 10)

	xlabel('Sample index', labelpad = 11)
	ax.xaxis.set_label_position('top')
	ylabel('Sample index')
	ax.tick_params(top = True, labeltop = True, bottom = False, labelbottom = False)
	gca().xaxis.set_major_locator(ticker.MultipleLocator(6))
	gca().yaxis.set_major_locator(ticker.MultipleLocator(6))
	grid(False)
	fig.savefig(f'output/{filename}.pdf', dpi = 1000)
	close(fig)	

def plot_unicalib_full_range(ax):
	
	mastr, bstr, cstr = unicalib_rounding()
	Tmin, Tmax, D47min, D47max = -5, 41, .54, .73
	Xmin, Xmax = (Tmax + 273.15)**-2, (Tmin + 273.15)**-2

	sca(ax)
	ax.grid(False)

	xi = linspace((2000+273.15)**-2, (-7+273.15)**-2)**.5

	ogls_2023.invT_xaxis()
	ogls_2023.plot_bff_ci(xi = xi, color = yello, alpha = 1)
	ogls_2023.plot_error_bars(ecolor = [.75]*3, elinewidth = 2/3, capthick = 2/3, capsize = 2)
	ogls_2023.plot_bff(color = 'k', xi = xi, lw = .75, label = f'Combined regression:\n$Δ_{{47}} = {mastr}\\cdot10^3\\,/\\,T^2 {bstr}\\,/\\,T + {cstr}$\n')
	plot([], [], '-', color = yello, lw = 4, label = '95 % confidence band')


	legs = {}

	for calib in calibs:
		if calib == devils_laghetto_2023:
			continue
		legs[calibs[calib]['label'].split(' ')[0]], = calib.plot_data(
			mec = 'k',
			marker = calibs[calib]['marker'],
			ms = calibs[calib]['ms'],
			mew = 2/3,
# 			label = calibs[calib]['label'].replace('-', '}$-$\\mathcal{'),
			zorder = calibs[calib]['zorder'] if 'zorder' in calibs[calib] else 10,
			)


	curaxis = list(axis())

	box_x1 = Xmin
	box_x2 = Xmax
	box_y1 = D47min
	box_y2 = D47max
	arrow_x = Xmin + (Xmax - Xmin)*0.25
	arrow_y = D47max + 0.1

	plot(
		[arrow_x, box_x1, box_x1, box_x2, box_x2, arrow_x, arrow_x],
		[box_y2, box_y2, box_y1, box_y1, box_y2, box_y2, arrow_y],
		'k-', alpha = .2, lw = 1.5, scaley = False, scalex = False, clip_on = False,
		)
	plot(arrow_x, arrow_y, '^', ms = 6, mec = 'None', mfc = [.8]*3, scaley = False, scalex = False, clip_on = False)

# 	atxt = f'{ogls_2023.bfp["a2"]/1000:.2f}\\cdot 10^3'
# 	btxt = f'{ogls_2023.bfp["a1"]:.2f}'
# 	ctxt = f'{ogls_2023.bfp["a0"]:.4f}'
# 
# 	txt = f'$Δ_{{47}}~=~ {ctxt} {btxt}\\cdot T^{{-1}} + {atxt}\\cdot T^{{-2}}$'
# 
# 	text(.95, .05, txt, va = 'bottom', ha = 'right', size = 10, color = 'k', transform = ax.transAxes)

	# legend_order = sorted([k for k in leg], key = lambda k: -len(leg[k].get_label()))

	xlabel('$1\,/\,T^2$', labelpad = 10.0)
	ylabel('Δ$_{47}$ (‰, I-CDES)', labelpad = 10.0)

	curaxis[-1] = .78
	axis(curaxis)

	legend(
		loc = 'lower right',
		fontsize = 9,
		bbox_to_anchor = (0.97, 0.03),
	# 	numpoints=1,
		frameon=False,
		labelspacing = -0.4,
		handlelength = 2.5,
		)


def plot_unicalib_low_T(ax2):

	Tmin, Tmax, D47min, D47max = -5, 41, .54, .73
	Xmin, Xmax = (Tmax + 273.15)**-2, (Tmin + 273.15)**-2

	sca(ax2)
	grid(False)
	xi = linspace(Xmin, Xmax)**.5
	ogls_2023.invT_xaxis(Ti = [0,15,35])
	ogls_2023.plot_bff_ci(xi = xi, color = yello, alpha = 1)
	ogls_2023.plot_error_bars(ecolor = [.75]*3, elinewidth = 2/3, capthick = 2/3, capsize = 2)
	lg_bff, = ogls_2023.plot_bff(color = 'k', lw = .75, label = 'Combined regression')
	lg_cl, = plot([], [], '-', color = yello, lw = 4, label = '95 % confidence band')

	lgs = []
	for calib in [
		anderson_2021_lsce,
		fiebig_2021,
		anderson_2021_mit,
		huyghe_2022,
		peral_2018,
		jautzy_2020,
		breitenbach_2018,
		]:
		if calib == devils_laghetto_2023:
			continue
		_, = calib.plot_data(
			mec = 'k',
			marker = calibs[calib]['marker'],
			ms = calibs[calib]['ms'],
			mew = 2/3,
			label = calibs[calib]['label'].replace('-', '}$-$\\mathcal{'),
			zorder = calibs[calib]['zorder'] if 'zorder' in calibs[calib] else 10,
			)
		lgs.append(_)

	axis([xi[0]**2, xi[-1]**2, D47min, D47max])
	xlabel('')
	ylabel('Δ$_{47}$ (‰, I-CDES)', labelpad = 6.0)
	gca().yaxis.set_major_locator(ticker.MultipleLocator(.05))

	lg = legend(
		lgs,
		[_.get_label() for _ in lgs],
		loc = 'upper left',
		fontsize = 9,
		bbox_to_anchor = (0.015, 0.985),
	# 	numpoints=1,
		frameon=False,
		labelspacing=0.4,
		handlelength = .5,
		)
	
	legend(
		[lg_bff, lg_cl],
		[_.get_label() for _ in [lg_bff, lg_cl]],
		loc = 'lower right',
		fontsize = 9,
		bbox_to_anchor = (0.97, 0.03),
	# 	numpoints=1,
		frameon=False,
		labelspacing=0.4,
		handlelength = 2.5,
		)
	
	ax2.add_artist(lg)

def plot_unicalib_residuals(ax3):

	sca(ax3)
	grid(False)
	ogls_2023.regress()

	x,y = zip(*[[r,k] for k,r in enumerate(sorted(ogls_2023.cholesky_residuals))])
	x = array([-10, *x, 10])
	y = array([0, *y, len(y)]) / len(y)

	plot(x, y, 'k-', lw = 1, drawstyle = 'steps')

	pvalue = kstest(ogls_2023.cholesky_residuals, 'norm', (0, 1)).pvalue

	text(.95, .05, f'N = {ogls_2023.N}', va = 'bottom', ha = 'right', size = 9, transform = gca().transAxes)
	text(.05, .95, f'p = {pvalue:{".2f" if pvalue>=0.1 else ".3f"}}', va = 'top', ha = 'left', size = 9, transform = gca().transAxes)
	text(.5, 1, 'Kolmogorov–Smirnov test\n', va = 'center', ha = 'center', size = 9, linespacing = 1.8, transform = gca().transAxes)



	x = linspace(-4.6, 4.6)
	y = norm().cdf(x)

	plot(x,y,'-', lw = 1.5, zorder = -10, color = yello)

	yticks([0, 1])
	axis([x[0], x[-1], -.05, 1.05])

	gca().xaxis.set_major_locator(ticker.MultipleLocator(2))
	gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'${x:+.0f}$' if x else '$0$'))

	xlabel('Cholesky residuals', size = 9)
	ylabel('Cumulative\ndistribution', labelpad = -8, size = 9)

def unicalib_plot():

	fig = figure(figsize = (5,8))
	subplots_adjust(
		left = 0.14,
		right = .96,
		top = .975,
		bottom = 0.07,
		wspace = -.9,
		hspace = .1)

	ax = subplot(212)
	ax2 = subplot(211)
	ax3 = fig.add_axes((.225, .33,.27,.135))

	plot_unicalib_full_range(ax)
	plot_unicalib_low_T(ax2)
	plot_unicalib_residuals(ax3)
	
	savefig('output/fig-11.pdf')
	close(fig)


def plot_95CL(sp = (121, 122), spl = 'ab'):

	calib = ogls_2023

	Ti = [1000, 200, 50, 0]
	X = linspace(1273.15**-2, 260**-2, 101)**.5
	T = 1/X - 273.15
	D47, sD47 = calib.T47(T = T, error_from = 'calib')
	_, sT = ogls_2023.T47(D47 = D47, error_from = 'calib')

	ax1 = subplot(sp[0], xlim = (.1, None))
	ax1.set_xscale('function', functions=(lambda T: (T+273.15)**-2, lambda X: X**-.5 - 273.15))
	ax2 = subplot(sp[1], sharex = ax1)
	
	sca(ax1)
# 	plot(T, sD47 * pfactor * 1000, '-', color = yello, lw = 2)
	plot(T, sD47 * pfactor * 1000, 'k-', dashes = (6,1,2,1), lw = 1.5)
	axis([Ti[-1], Ti[0], 1.7, 4.3])
	ylabel('95 % confidence\non Δ$_{47}$ (ppm)')
	ax1.yaxis.set_major_formatter(FuncFormatter(lambda x,l: f'±$\\,${x:.0f}' if x else '0'))
	text(.97, .96, spl[0], weight = 'bold', size = 10, va = 'top', ha = 'right', transform = ax1.transAxes)
	
	sca(ax2)
# 	semilogy(T, sT * pfactor, '-', color = yello, lw = 2)
	semilogy(T, sT * pfactor, 'k-', dashes = (6,1,2,1), lw = 1.5)
	xlabel('T (°C)')
	ylabel('95 % confidence\non T estimate (°C)')
	ax2.yaxis.set_major_formatter(FuncFormatter(lambda x,l: f'±$\\,${x:.0f}' if x else '0'))
	axis([Ti[-1], Ti[0], 0.3, None])
	text(.97, .96, spl[1], weight = 'bold', size = 10, va = 'top', ha = 'right', transform = ax2.transAxes)

	xticks(Ti)


def KS_plots(add_arrow = True):

	fig = figure(figsize = (6,6))
	subplots_adjust(
		left = .05,
		right = .95,
		bottom = .1,
		top = .95,
		hspace = .5,
		wspace = .3,
		)

	pvalues = []

	moz = fig.subplot_mosaic(\
		'''
		..Pp
		AaPp
		AaBb
		FfBb
		FfHh
		JjHh
		JjDd
		..Dd
		''')

	for calib in calibs:
		if calib == anderson_2021_lsce:
			continue

		_calib_ = calibs[calib]
		
		ax1 = moz[_calib_['moz']]
		ax1.grid(False)
		sca(ax1)

		if calib.T.max() > 100:
			Tmin, Tmax = -5, 1800
			Ti = [0,100,1000]
		else:
			Tmin, Tmax = -5, 55
			Ti = [0,20,50]

		xpi = linspace((Tmax+273.15)**-2, (Tmin+273.15)**-2, 101)
		xi = xpi**.5	

	# 	xticks([])
		yticks([])
		calib.plot_error_bars(ecolor = [.75]*3, elinewidth = 2/3, capthick = 2/3, capsize = 0, zorder = -300)
		calib.plot_data(
			mec = 'k',
			marker = _calib_['marker'],
			ms = _calib_['ms']*0.7,
			mew = 2/3,
	# 		mfc = yello,
			)

	# 	_axis_ = axis()

	# 	c.plot_bff_ci(xi = xi, color = yello, alpha = 1, zorder = -200)
		ogls_2023.plot_bff(xi = xi, lw = .75, zorder = -100, color = [.8]*3)

		axis([xpi[0], xpi[-1], None, None])
		text(.05, .93, _calib_['label'].split(' ')[0], va = 'top', ha = 'left', transform = ax1.transAxes, size = 8)
# 		text(.95, .05, '('+_calib_['label'].split('(')[1], va = 'bottom', ha = 'right', transform = ax1.transAxes, size = 7)

		calib.invT_xaxis(Ti = Ti)
		if _calib_['moz'] not in 'JD':
			xlabel('')
			ax1.tick_params(length = 2, labelbottom = False)
		else:
			xlabel(r'$1\,/\,T^2$', labelpad = 8)

		ax1.tick_params(axis='both', which='major', labelsize=8)
		ylabel('Δ$_{47}$')

		if add_arrow:
			x_arrow, y_arrow = (xpi[-1]*1.15, .6) if Tmax > 100 else (xpi[-1]*1.05, .66)
			plot([xpi[-1], x_arrow], [y_arrow, y_arrow], '-', color = 'k', lw = 1, clip_on = False)
			plot(x_arrow, y_arrow, '>', mfc = 'k', mec = 'k', mew = 0, ms = 4.5, clip_on = False)


		ax2 = moz[_calib_['moz'].lower()]
		ax2.tick_params(axis='both', which='major', labelsize=8)
		ax2.grid(False)
		sca(ax2)

		_c_ = D47calib(
			samples = calib.samples,
			T = calib.T,
			D47 = calib.D47,
			sT = calib.sT,
			sD47 = calib.sD47,
			degrees = ogls_2023.degrees,
			)

		normalized_residuals = sorted(_c_.cost_fun(ogls_2023.bfp)[:,0])
		x,y = zip(*[[r,k] for k,r in enumerate(normalized_residuals)])
		y = [0]+list(y)+[len(y)]
		x = [-10]+list(x)+[10]
		x = array(x)
		y = array(y)/(len(y)-2)
		plot(x, y, 'k-', lw = 1, drawstyle = 'steps')

		pvalue = kstest(x[1:-1], 'norm', (0, 1)).pvalue
		
		if calib != devils_laghetto_2023:
			pvalues += [pvalue]

		text(.95, .05, f'p = {pvalue:{".2f" if pvalue>=0.05 else ".3f"}}', va = 'bottom', ha = 'right', size = 8, transform = gca().transAxes)

		text(.05, .95, f'N = {len(normalized_residuals)}', va = 'top', ha = 'left', size = 8, transform = gca().transAxes)

		x = linspace(-4.6, 4.6)
		y = norm().cdf(x)

		plot(x,y,'-', lw = 1.5, zorder = -10, color = yello)

		yticks([0, 1])

		ax2.xaxis.set_major_locator(ticker.MultipleLocator(3))
		ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'${x:+.0f}$' if x else '$0$'))

		axis([x[0], x[-1], -.05, 1.05])
		xlabel('Cholesky residuals', size = 9)
		if _calib_['moz'] not in 'JD':
			xlabel('')
			ax2.tick_params(length = 2, labelbottom = False)
		ylabel('CDF', labelpad = -8, size = 9)


# 		ax2.add_artist(mpatches.FancyArrow(
# 			x = -.21,
# 			y = 0.7,
# 			dx = .04,
# 			dy = 0,
# 			width = 0.1,
# 			head_width = 0.15,
# 			head_length = 0.08,
# 			clip_on = False,
# 			transform = ax2.transAxes,
# 			fc = [_*.5+.5 for _ in yello],
# 			ec = [.5,.5,.5],
# 			))


	savefig(f'output/fig-14.pdf')
	close(fig)

	fig = figure(figsize = (3,3))
	subplots_adjust(.15,.2,.9,.95)

	pvalues = sorted(pvalues)
	x,y = zip(*[[r,k] for k,r in enumerate(pvalues)])
	y = [0]+list(y)+[len(y)]
	x = [-10]+list(x)+[10]
	x = array(x)
	y = array(y)/(len(y)-2)
	plot(x, y, 'k-', lw = 1, drawstyle = 'steps')

	pvalue = kstest(pvalues, 'uniform', (0, 1)).pvalue
	text(.95, .05, f'p = {100*pvalue:{".0f" if pvalue>0.1 else ".1f"}} %', va = 'bottom', ha = 'right', size = 10, transform = gca().transAxes)

	text(.05, .95, f'N = {len(pvalues)}', va = 'top', ha = 'left', size = 10, transform = gca().transAxes)

	x = array([0, 1])
	y = x

	plot(x,y,'-', lw = 1.5, zorder = -10, color = yello)

	grid(False)
	yticks([0, 1])
	xticks([0, 1])
	axis([x[0], x[-1], -.05, 1.05])
	xlabel('p-value')
	ylabel('CDF', labelpad = -5)

	savefig(f'output/fig-S2.pdf')
	close(fig)


def calib_plots():

	calibs = {
		devils_laghetto_2023: dict(
			marker = 'o',
			ms = 4,
			zorder = 200,
			title = '“Devils Laghetto” calibration,\ncombined from $\\mathcal{[A21·LSCE]}$ and $\\mathcal{[F21]}$',
			label = 'DL23',
			filename = 'fig-18.pdf',
			ls = (0, (6,2)),
			Ti = [35,25,15,6],
			),
		anderson_2021_lsce: dict(
			marker = 'o',
			ms = 4,
			zorder = 200,
			title = 'Anderson et al. [2021] – LSCE data',
			label = 'A21_LSCE',
			filename = 'fig-S3.pdf',
			ls = (0, (6,2)),
			Ti = [35,25,15,6],
			),
		anderson_2021_mit: dict(
			marker = 'd',
			ms = 4,
			title = 'Anderson et al. [2021] – MIT data',
			label = 'A21_MIT',
			filename = 'fig-17d.pdf',
			ls = (0, (6,2)),
			),
		breitenbach_2018: dict(
			marker = 'v',
			ms = 4,
			title = 'Breitenbach et al. [2018]',
			label = 'B18',
			filename = 'fig-17a.pdf',
			ls = (0, (6,2,2,2)),
			Ti = [50,30,15,0],
			),
		fiebig_2021: dict(
			marker = 's',
			ms = 3.6,
			title = 'Fiebig et al. [2021]',
			label = 'F21',
			filename = 'fig-17e.pdf',
			ls = (0, (6,2,2,2)),
	# 		Ti = [40,25,10,0],
			),
		huyghe_2022: dict(
			marker = hdiamond,
			ms = 4,
			title = 'Huyghe et al. [2022]',
			label = 'H22',
			filename = 'fig-17f.pdf',
			ls = (0, (2,1)),
			Ti = [25,15,5,-3],
			),
		jautzy_2020: dict(
			marker = 'D',
			ms = 3.2,
			title = 'Jautzy et al. [2020]',
			label = 'J20',
			filename = 'fig-17c.pdf',
			ls = (0, (6,2,2,2,2,2)),
			),
		peral_2018: dict(
			marker = '^',
			ms = 4,
			title = 'Peral et al. [2018]',
			label = 'P18',
			filename = 'fig-17b.pdf',
			ls = (0, (6,2,2,2,2,2)),
			Ti = [25,17,8,1],
			),
		}

	for calib in calibs:
		_calib_ = calibs[calib]
		XsX = list(zip(calib.X, diag(calib.sX)**.5))
		label = _calib_['label']
		filename = _calib_['filename']
		fig = figure(figsize = (4,4))
		subplots_adjust(.15,.15,.9,.9)
		calib.invT_xaxis(**({'Ti': _calib_['Ti']} if 'Ti' in _calib_ else {}))
		calib.plot_error_bars(ecolor = [0]*3, elinewidth = .8, capthick = 0.8, capsize = 2)

		calib.plot_bff_ci(span = 1.1, color = yello, lw = 0, alpha = .5, xpmin = 2000**-2)
		calib.plot_bff(span = 1.1, ls = '-', color = 'k', lw = 1, xpmin = 2000**-2)

		calib.plot_data(marker = _calib_['marker'], ms = _calib_['ms'], mfc = 'w', mec = 'k', mew = .8)
		title(_calib_['title'], size = 9 if 'Devils Laghetto' in _calib_['title'] else 10)
		text(.93, .07,
			f"[{label.replace('_', '·')}]",
			font = 'JetBrains Mono',
			weight = 'bold',
			va = 'bottom',
			ha = 'right',
			transform = gca().transAxes,
			size = 10,
			bbox = dict(
				boxstyle = 'round',
				pad = .4,
				ec = [.85]*3,
				fc = [.95]*3,
				)
			)
		ylabel('Δ$_{47}$ (‰, I-CDES)')
		grid(False)
		if axis()[0] < (400)**-2:
			gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
		else:
			gca().yaxis.set_major_locator(ticker.MultipleLocator(0.04))
		savefig(f"output/{filename}")


def unicalib_rounding():
	a,b,c = ogls_2023.bfp['a2'], ogls_2023.bfp['a1'], ogls_2023.bfp['a0']
	# D47 = a/T**2 + b/T + c
# 	1/T = (-b + (b**2 - 4a(c-D47))**.5)/(2a)
# 	1/T = -b/a/2 + (b**2/a**2/4 - (c-D47)/a)**.5
# 	1/T = -b/a/2 + (D47/a + b**2/a**2/4 - c/a)**.5

	ma = a / 1e3

	A = -b/2/a
	B = 1/a
	C = b**2/4/a**2 - c/a
	# 1/T = A + (B * D47 + C)**0.5

	kA = 1e3 * A
	MB = 1e6 * B
	MC = 1e6 * C

	T = linspace(-10, 1200, 1211) + 273.15
	D47 = a/T**2 + b/T + c

	# print()
	persist, k = True, 0
	while persist:
		_ma_ = round(ma,k)
		dD47 = D47 - (1e3 * _ma_ / T**2 + b / T + c)
	# 	print(abs(dD47).max())
		persist = abs(dD47).max() > 0.0001
		k += 1

	# print()
	persist, k = True, 0
	while persist:
		_b_ = round(b,k)
		dD47 = D47 - (1e3 * ma / T**2 + _b_ / T + c)
	# 	print(abs(dD47).max())
		persist = abs(dD47).max() > 0.0001
		k += 1

	# print()
	persist, k = True, 0
	while persist:
		_c_ = round(c,k)
		dD47 = D47 - (1e3 * ma / T**2 + b / T + _c_)
	# 	print(abs(dD47).max())
		persist = abs(dD47).max() > 0.0001
		k += 1

	# print()
	persist, k = True, 0
	while persist:
		_kA_ = round(kA,k)
		dT = T - (1000 / (_kA_ + (MB*D47 + MC)**.5))
	# 	print(abs(dT).max())
		persist = abs(dT).max() > 0.1
		k += 1

	# print()
	persist, k = True, 0
	while persist:
		_MB_ = round(MB,k)
		dT = T - (1000 / (kA + (_MB_*D47 + MC)**.5))
	# 	print(abs(dT).max())
		persist = abs(dT).max() > 0.1
		k += 1

	# print()
	persist, k = True, 0
	while persist:
		_MC_ = round(MC,k)
		dT = T - (1000 / (kA + (MB*D47 + _MC_)**.5))
	# 	print(abs(dT).max())
		persist = abs(dT).max() > 0.1
		k += 1

	rmswd = {}
	for calib in [
		breitenbach_2018,
		peral_2018,
		jautzy_2020,
		anderson_2021_mit,
		anderson_2021_lsce,
		fiebig_2021,
		huyghe_2022,
		devils_laghetto_2023,
		]:
		_calib_ = D47calib(
			samples = calib.samples,
			T = calib.T,
			D47 = calib.D47,
			sT = calib.sT,
			sD47 = calib.sD47,
			degrees = ogls_2023.degrees,
			bfp = ogls_2023.bfp,
			)
		residuals = _calib_.cost_fun(ogls_2023.bfp)
		rmswd[calib] = (residuals**2).mean()**.5

	libexport.write(f'''
\\define@key{{py}}{{ogls_2023_ma}}[0]{{{_ma_}}}
\\define@key{{py}}{{ogls_2023_b_sign}}[0]{{{str(_b_)[0]}}}
\\define@key{{py}}{{ogls_2023_b}}[0]{{{abs(_b_)}}}
\\define@key{{py}}{{ogls_2023_c}}[0]{{{_c_}}}
\\define@key{{py}}{{ogls_2023_rmswd}}[0]{{{ogls_2023.red_chisq**.5:.2f}}}
\\define@key{{py}}{{ogls_2023_kspvalue}}[0]{{{ogls_2023.ks_pvalue:.2f}}}

\\define@key{{py}}{{ogls_2023_kA}}[0]{{{_kA_}}}
\\define@key{{py}}{{ogls_2023_MB}}[0]{{{_MB_}}}
\\define@key{{py}}{{ogls_2023_MC}}[0]{{{_MC_}}}

\\define@key{{py}}{{B18_a0}}[0]{{{breitenbach_2018.bfp['a0']:.3f}}}
\\define@key{{py}}{{B18_a1}}[0]{{}}
\\define@key{{py}}{{B18_a2}}[0]{{{breitenbach_2018.bfp['a2']/1000:.2f}}}
\\define@key{{py}}{{B18_irmswd}}[0]{{{breitenbach_2018.red_chisq**.5:.2f}}}
\\define@key{{py}}{{B18_rmswd}}[0]{{{rmswd[breitenbach_2018]:.2f}}}

\\define@key{{py}}{{P18_a0}}[0]{{{peral_2018.bfp['a0']:.3f}}}
\\define@key{{py}}{{P18_a1}}[0]{{}}
\\define@key{{py}}{{P18_a2}}[0]{{{peral_2018.bfp['a2']/1000:.2f}}}
\\define@key{{py}}{{P18_irmswd}}[0]{{{peral_2018.red_chisq**.5:.2f}}}
\\define@key{{py}}{{P18_rmswd}}[0]{{{rmswd[peral_2018]:.2f}}}

\\define@key{{py}}{{J20_a0}}[0]{{{jautzy_2020.bfp['a0']:.3f}}}
\\define@key{{py}}{{J20_a1}}[0]{{{jautzy_2020.bfp['a1']:.2f}}}
\\define@key{{py}}{{J20_a2}}[0]{{{jautzy_2020.bfp['a2']/1000:.2f}}}
\\define@key{{py}}{{J20_irmswd}}[0]{{{jautzy_2020.red_chisq**.5:.2f}}}
\\define@key{{py}}{{J20_rmswd}}[0]{{{rmswd[jautzy_2020]:.2f}}}

\\define@key{{py}}{{A21_a0}}[0]{{{anderson_2021_mit.bfp['a0']:.3f}}}
\\define@key{{py}}{{A21_a1}}[0]{{}}
\\define@key{{py}}{{A21_a2}}[0]{{{anderson_2021_mit.bfp['a2']/1000:.2f}}}
\\define@key{{py}}{{A21_irmswd}}[0]{{{anderson_2021_mit.red_chisq**.5:.2f}}}
\\define@key{{py}}{{A21_rmswd}}[0]{{{rmswd[anderson_2021_mit]:.2f}}}

\\define@key{{py}}{{A21b_a0}}[0]{{{anderson_2021_lsce.bfp['a0']:.3f}}}
\\define@key{{py}}{{A21b_a1}}[0]{{}}
\\define@key{{py}}{{A21b_a2}}[0]{{{anderson_2021_lsce.bfp['a2']/1000:.2f}}}
\\define@key{{py}}{{A21b_irmswd}}[0]{{}}
\\define@key{{py}}{{A21b_rmswd}}[0]{{{rmswd[anderson_2021_lsce]:.2f}}}

\\define@key{{py}}{{F21_a0}}[0]{{{fiebig_2021.bfp['a0']:.3f}}}
\\define@key{{py}}{{F21_a1}}[0]{{{fiebig_2021.bfp['a1']:.2f}}}
\\define@key{{py}}{{F21_a2}}[0]{{{fiebig_2021.bfp['a2']/1000:.2f}}}
\\define@key{{py}}{{F21_irmswd}}[0]{{{fiebig_2021.red_chisq**.5:.2f}}}
\\define@key{{py}}{{F21_rmswd}}[0]{{{rmswd[fiebig_2021]:.2f}}}

\\define@key{{py}}{{H22_a0}}[0]{{{huyghe_2022.bfp['a0']:.3f}}}
\\define@key{{py}}{{H22_a1}}[0]{{}}
\\define@key{{py}}{{H22_a2}}[0]{{{huyghe_2022.bfp['a2']/1000:.2f}}}
\\define@key{{py}}{{H22_irmswd}}[0]{{{huyghe_2022.red_chisq**.5:.2f}}}
\\define@key{{py}}{{H22_rmswd}}[0]{{{rmswd[huyghe_2022]:.2f}}}

\\define@key{{py}}{{DL23_a0}}[0]{{{devils_laghetto_2023.bfp['a0']:.3f}}}
\\define@key{{py}}{{DL23_a1}}[0]{{}}
\\define@key{{py}}{{DL23_a2}}[0]{{{devils_laghetto_2023.bfp['a2']/1000:.2f}}}
\\define@key{{py}}{{DL23_irmswd}}[0]{{{devils_laghetto_2023.red_chisq**.5:.2f}}}
\\define@key{{py}}{{DL23_rmswd}}[0]{{{rmswd[devils_laghetto_2023]:.2f}}}
''')

	return _ma_, _b_, _c_

def compare_with_theory():
	calib = ogls_2023

	R13_VPDB = 0.01118      # (Chang & Li, 1990)
	R18_VSMOW = 0.0020052   # (Baertschi, 1976)
	LAMBDA_17 = 0.528       # (Barkan & Luz, 2005)
	R17_VSMOW = 0.00038475  # (Assonov & Brenninkmeijer, 2003, rescaled to R13_VPDB)
	R18_VPDB = R18_VSMOW * 1.03092
	R17_VPDB = R17_VSMOW * 1.03092 ** LAMBDA_17

	### SCHAUBLE 2006
	Keq3668 = lambda x: (
		- 3.40752e6  * x**4
		+ 2.36545e4  * x**3
		- 2.63167    * x**2
		- 5.85372e-3 * x
		+ 1
		)

	Keq2678 = lambda x: (
		- 2.81387e5  * x**4
		+ 3.26769e3  * x**3
		- 3.98299    * x**2
		- 1.68454e-3 * x
		+ 1
		)

	R3668 = R13_VPDB * R18_VPDB * 3
	R2678 = R17_VPDB * R18_VPDB * 6
	x3668, x2678 = R3668/(R3668+R2678), R2678/(R3668+R2678)

	Keq63 = lambda x: x3668 * Keq3668(x) + x2678 * Keq2678(x)
	D63eq_schauble = lambda x: (Keq63(x) - 1)*1000

	### DATA REPORTED BY HILL 2020, ORIGINALLY FROM HILL 2014:
	Hill_et_al_2020_calcite_predicitons = '''
T	D63	D64	D65
0	0.4701493	0.1556794	1.1311752
10	0.4378408	0.1399296	1.0486074
20	0.4080429	0.1259686	0.9730013
22	0.4023646	0.1233712	0.9586546
25	0.3940150	0.1195893	0.9375944
30	0.3805332	0.1135773	0.9036807
40	0.3551131	0.1025649	0.8400466
50	0.3316046	0.0927648	0.7815670
60	0.3098476	0.0840315	0.7277676
70	0.2896977	0.0762380	0.6782245
80	0.2710240	0.0692734	0.6325578
90	0.2537077	0.0630407	0.5904261
100	0.2376405	0.0574548	0.5515222
200	0.1276123	0.0245551	0.2902025
300	0.0725627	0.0118753	0.1630424
400	0.0435056	0.0063393	0.0970151
500	0.0273455	0.0036587	0.0606693
600	0.0179112	0.0022467	0.0395948
700	0.0121580	0.0014503	0.0268042
800	0.0085114	0.0009751	0.0187254
900	0.0061204	0.0006782	0.0134419
1000	0.0045050	0.0004852	0.0098798
'''[1:-1].split('\n')
	Hill_et_al_2020_calcite_predicitons = [l.split('\t') for l in Hill_et_al_2020_calcite_predicitons[1:]]
	Hill_et_al_2020_calcite_predicitons = array(Hill_et_al_2020_calcite_predicitons, dtype = float)
	_T, _D63eq, _D64eq, _D65eq = Hill_et_al_2020_calcite_predicitons.T
	
	_X = (_T+273.15)**-1
	
	_XX = np.vstack((_X ** 4, _X ** 3, _X ** 2, _X)).T
	a_hill2020, b_hill2020, c_hill2020, d_hill2020 = linalg.lstsq(_XX, _D63eq)[0]
	
	D63eq_hill = lambda x: (
		a_hill2020 * x**4
		+ b_hill2020 * x**3
		+ c_hill2020 * x**2
		+ d_hill2020 * x
		)

	### IDENTICAL FIT REPORTED BY FIEBIG 2021
# 	D63eq_hill = lambda x: (
# 		- 5.897 * x
# 		- 3.521e3 * x**2
# 		+ 2.391e7 * x**3
# 		-3.541e9 * x**4
# 		)

# 	f_theory = lambda x: 1.035 * D63eq(x)
# 
# 	# Guo et al. (2009):
# 	f_theory_2 = lambda x: -3.33040e9*x**4 + 2.32415e7*x**3 - 2.91282e3*x**2 - 5.54042*x + 0.23252 - 0.088

	D63eq = D63eq_hill

	D63 = D63eq((calib.T+273.15)**-1)
	e = 0.001
	J_D63 = diag([(
		D63eq(((calib.T[k]+e*calib.sT[k,k]**.5)+273.15)**-1)
		- D63eq(((calib.T[k]-e*calib.sT[k,k]**.5)+273.15)**-1)
		)/(2*e) for k in range(calib.N)])
	CM_D63 = J_D63 @ calib.sT @ J_D63

	offsetreg = ogls.Polynomial(
		X = D63, sX = CM_D63,
		Y = calib.D47, sY = calib.sD47,
		degrees = [0, 1],
		)
	offsetreg.regress()
	a, b, CM = offsetreg.bfp['a1'], offsetreg.bfp['a0'], offsetreg.bfp_CM

	libexport.write(f'''
\\define@key{{py}}{{D47vsD63_slope}}[0]{{{a:.3f}~±~{pfactor * CM[1,1]**.5:.3f}}}
\\define@key{{py}}{{D47vsD63_intercept}}[0]{{{b:.3f}~±~{pfactor * CM[0,0]**.5:.3f}}}
''')

	fig = figure(figsize = (3.5,3.5))
	subplots_adjust(.18,.15,.95,.92)
	
	X = D63eq((calib.T+273.15)**-1)
	Y = calib.D47
	eY = pfactor * diag(calib.sD47)**.5
	eX = pfactor * diag(CM_D63)**.5
	xi = linspace(-.1,.6)
	yi = a*xi + b
	syi = (CM[1,1] * xi**2 + 2*CM[0,1]*xi + CM[0,0])**.5


# 	fill_between(xi, yi+pfactor*syi, yi-pfactor*syi, color = 'k', alpha = .15, lw = 1, label = 'foo')
	plot(xi, yi, 'k-', lw = 0.6, dashes = (6,2,2,2), label = f'''

$Δ_{{47}}$ = a $Δ_{{63}}$ + b
a = {a:.3f} ± {CM[1,1]**.5:.3f} (1SE)
b = {b:.3f} ± {CM[0,0]**.5:.3f} (1SE)''')
# 	errorbar(X, Y, eY, eX,
# 		ecolor = 'k', elinewidth = .8, capsize = 1, capthick = .8, fmt = 'None', alpha = .2,
# 		)
	plot(X, Y, 'ko',
		mec = 'k', mew = 0, ms = 4, alpha = 1/3,
		)
	
	xlabel('Theoretically predicted Δ$_{63}$ (‰)')
	ylabel('Observed Δ$_{47}$ (‰)')
	axis([-.04,.51,.16,.71])
# 	axis([-.07,.53,.13,.73])
	xticks([.0,.2,.4])
	yticks([.2,.4,.6])
	legend(handlelength = 3, frameon = False, fontsize = 8, loc = 'center left', bbox_to_anchor = (.02,.94))
	grid(None)

	savefig('output/fig-15.pdf')
	close(fig)


# 	Ti = (linspace(2000**-2, 270**-2, 1001))**-.5 - 273.15
#    
# 	ax = subplot(111)
# 	ax.set_xscale('function', functions=(lambda T: (T+273.15)**-2, lambda X: X**-.5 - 273.15))
# 	plot(Ti, Ti*0+offset)
# 	errorbar(offsetreg.T, offsetreg.D47, 1.96*diag(offsetreg.sD47)**.5,
# 		ecolor = 'r', elinewidth = 1, capsize = 2, capthick = 1, fmt = 'None',
# 		)
# 	plot(offsetreg.T, offsetreg.D47, 'wo',
# 		mec = 'r', mew = 1, ms = 5,
# 		)


# 	plot(Ti, 1.035*D63eq((Ti+273.15)**-1), label = 'Scaled Δ$_{63}$ prediction')
# 	plot(Ti, offset + 1.035*D63eq((Ti+273.15)**-1))
# # 	plot(Ti, f_theory((Ti+273.15)**-1), label = 'Theory')
# # 	plot(Ti, f_theory_2((Ti+273.15)**-1), label = 'Guo')
# 	errorbar(calib.T, calib.D47, 1.96*diag(calib.sD47)**.5,
# 		ecolor = 'r', elinewidth = 1, capsize = 2, capthick = 1, fmt = 'None',
# 		)
# 	plot(calib.T, calib.D47, 'wo', label = 'Δ$_{47}$ observations',
# 		mec = 'r', mew = 1, ms = 5,
# 		)

# 	axis([-5, 2000, None, None])
# 	xticks([0,25,50,100,200,400,1000])
# 	legend()
	
# 	show()

def conf_limits():
	fig = figure(figsize = (4,4))
	subplots_adjust(.15,.15,.95,.95)
	ax = subplot(111, xlim = (.1, None))
	ax.set_xscale('function', functions=(lambda T: (T+273.15)**-2, lambda X: X**-.5 - 273.15))

	X = (linspace(2000**-2, 260**-2, 101))**.5
	T = 1/X - 273.15
	Y = ogls_2023.bff(X)
	Ymin = Y*0-1
	Ymax = Y*0+1
	for calib in calibs:
		if calib == devils_laghetto_2023:
			continue
		Ymax = array([Ymax, calib.bff(X) + pfactor * calib.bff_se(X) - ogls_2023.bff(X)]).min(0)
		Ymin = array([Ymin, calib.bff(X) - pfactor * calib.bff_se(X) - ogls_2023.bff(X)]).max(0)
		Xc = calib._xlinspace(span = 1)[0]
		plot(
			Xc**-1 - 273.15,
			1000*(calib.bff(Xc) + pfactor * calib.bff_se(Xc) - ogls_2023.bff(Xc)),
			'k-', dashes = (3,2), lw = 0.75, zorder = 200)
		plot(
			Xc**-1 - 273.15,
			1000*(calib.bff(Xc) - pfactor * calib.bff_se(Xc) - ogls_2023.bff(Xc)),
			'k-', dashes = (3,2), lw = 0.75, zorder = 200)
	fill_between(T, 1000*Ymax, 1000*Ymin, color = yello, lw = 0, alpha = .5)
	axhline(0, color = 'k', lw = 0.75, zorder = 300)
# 	fill_between(T, 1000 * pfactor * ogls_2023.bff_se(X), -1000 * pfactor * ogls_2023.bff_se(X), color = 'k', lw = 0, alpha = .2)
	xticks([1200, 400, 200, 100, 50, 20, 0])
	axis([-4, 1200, None, None])
	xlabel('T (°C)')
	ylabel('Δ$_{47}$ difference from $\\mathcal{[OGLS23]}$ (ppm)')
	ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'${x:+.0f}$' if x else '$0$'))
	savefig('output/fig-S1.pdf')
	close(fig)

def cov_ellipse(cov, q = .95):
	"""
	Parameters
	----------
	cov : (2, 2) array
		Covariance matrix.
	q : float
		Confidence level, should be in (0, 1)

	Returns
	-------
	width, height, rotation :
		 The lengths of two axises and the rotation angle in degree
	for the ellipse.
	"""

	from scipy.stats import chi2

	r2 = chi2.ppf(q, 2)
	val, vec = linalg.eigh(cov)
	width, height = 2 * sqrt(val[:, None] * r2)
	rotation = degrees(arctan2(*vec[::-1, 0]))

	return width, height, rotation

def conf_ellipses():
	
	from matplotlib.patches import Ellipse
	T0 = 25.

	fig = figure(figsize = (4,5))
	ax = subplot(111)
	
	kcolor = -1
	for calib in [
		breitenbach_2018,
		peral_2018,
		jautzy_2020,
		anderson_2021_mit,
		anderson_2021_lsce,
		fiebig_2021,
		huyghe_2022,
		]:
		if calib == devils_laghetto_2023:
			continue
		if 'a1' in calib.bfp:

			# x = T**-2
			# intercept = a0 + a1 * x**0.5 + a2 * x
			# slope = 0.5 * a1 * x-**0.5 + a2
			
			X0 = (T0+273.15)**-2
			J0 = array([
				[0, 0.5 * X0**-.5, 1],
				[1, X0**.5, X0],
				])
			
			CM0 = J0 @ calib.bfp_CM @ J0.T
			CM0[0,0] /= 1e6
			CM0[0,1] /= 1e3
			CM0[1,0] /= 1e3
			
			A = (0.5 * calib.bfp['a1'] * X0**-0.5 + calib.bfp['a2'])/1000
			B0 = calib.bfp['a0'] + calib.bfp['a1'] * X0**0.5 + calib.bfp['a2'] * X0

		else:

			A, B, CM = calib.bfp['a2']/1000, calib.bfp['a0'], calib.bfp_CM[::-1,:][:,::-1]*1.0
			X0 = 1000 * (T0+273.15)**-2
			CM[0,0] /= 1e6
			CM[0,1] /= 1e3
			CM[1,0] /= 1e3

			B0 = B+A*X0
			CM0 = array([[CM[0,0], CM[0,1] + X0 * CM[0,0]], [CM[0,1] + X0 * CM[0,0], CM[1,1] + 2*X0*CM[1,0] + CM[0,0]*X0**2]])
	
		kw = dict(ls = '-', marker = 'None', lw = 1, color = calibs[calib]['ecolor'])
		w,h,r = cov_ellipse(CM0, 0.95)
		plot([], [], label = calibs[calib]['label'], **kw)
		ax.add_artist(
			Ellipse(
				xy = (A, B0), width = w, height = h, angle = r,
				fc = 'None', ec = kw['color'], lw = kw['lw'], ls = '-', zorder = 100
				)
			)



	calib = ogls_2023
	
	X0 = (T0+273.15)**-2
	J0 = array([
		[0, 0.5 * X0**-.5, 1],
		[1, X0**.5, X0],
		])
	
	CM0 = J0 @ calib.bfp_CM @ J0.T
	CM0[0,0] /= 1e6
	CM0[0,1] /= 1e3
	CM0[1,0] /= 1e3
	
	A = (0.5 * calib.bfp['a1'] * X0**-0.5 + calib.bfp['a2'])/1000
	B0 = calib.bfp['a0'] + calib.bfp['a1'] * X0**0.5 + calib.bfp['a2'] * X0

	kw = dict(ls = '-', marker = 'None', lw = 1.5, color = 'k')
	w,h,r = cov_ellipse(CM0, 0.95)
# 	plot([], [], label = 'Combined calibration', **kw)
	ax.add_artist(
		Ellipse(
			xy = (A, B0), width = w, height = h, angle = r,
			fc = 'None', ec = kw['color'], lw = kw['lw'], ls = '-', zorder = 100, hatch='////',
			label = '$\\mathcal{[OGLS23]}$ (combined regression)',
			)
		)

	legend(
		fontsize = 8,
		loc = 'lower center',
		bbox_to_anchor = (0.5, 1.03),
# 		bbox_transform = fig.transFigure,
		)
	x0, y0 = 40, .600
	axis([x0-12, x0+12, y0-0.024, y0+0.024])
	ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(.01))
	xlabel('Regression slope / $10^3$')
	ylabel(f'Predicted $Δ_{{47}}$ value at {T0:.0f} °C')
	subplots_adjust(.16, .12, .95,.65)
	savefig('output/fig-13.pdf')
	close(fig)

def qmc(
	Ni = 2**16,
	calib = ogls_2023,
	reuse_old_qmc = True,
	Ti = [0., 25, 1000],
	binstep = 0.5e-3,
	plotwidth = 14e-3,
	seed = 47,
	sp = (131, 132, 133),
	spl = 'abc',
	):
	from scipy.stats.qmc import MultivariateNormalQMC
	from scipy.stats import norm
	from scipy.linalg import block_diag
	from tqdm import trange
	from csv import DictReader
	from pathlib import Path

	export_str('Nqmc', f'2^{{{log2(Ni):.0f}}}', libexport)
	
	if reuse_old_qmc:
		for csvfile in Path('.').glob('qmc_*.csv'):
			if int(csvfile.name[4:-4]) >= Ni:
				qmc_output = genfromtxt(csvfile, delimiter = ',')[:Ni]
				break			
		else:
			csvfile = f'qmc_{Ni}.csv'
			reuse_old_qmc = False

	if not reuse_old_qmc:
		csvfile = f'qmc_{Ni}.csv'
		dist = MultivariateNormalQMC(
			mean = [_ for _ in calib.D47] + [_ for _ in calib.T],
			cov = block_diag(calib.sD47, calib.sT),
			seed = seed,
			)
		sampled_input = dist.random(Ni) # sample.shape = (Ni, calib.N)
		sampled_output = zeros((Ni, len(calib.bfp)))
		
		with open(csvfile, 'w') as fid:
			for k in trange(Ni):
				_calib_ = D47calib(
					samples = calib.samples,
					T = sampled_input[k,-calib.N:],
					D47 = sampled_input[k,:calib.N],
					sT = calib.sT,
					sD47 = calib.sD47,
					degrees = calib.degrees,
					)
				if k > 0:
					fid.write('\n')
				fid.write(','.join([f'{_calib_.bfp[x]}' for x in calib.bfp]))

		qmc_output = genfromtxt(csvfile, delimiter = ',')

	D47 = {
		t: (
			qmc_output[:,0]
			+ qmc_output[:,1]/(t+273.15)
			+ qmc_output[:,2]/(t+273.15)**2
			)[:]
		for t in Ti
		}
		
	for k,t in enumerate(Ti):
		ax = subplot(sp[k])
		xmin = ceil((D47[t].mean() - plotwidth/2) / binstep) * binstep
		xmax = floor((D47[t].mean() + plotwidth/2) / binstep) * binstep
		Nb = int((xmax-xmin) / binstep) + 1
		for sty, alfa, lw in [('stepfilled', .15, 0), ('step', 1, 1.5)]:
			hist(
				D47[t],
				bins = linspace(xmin, xmax, Nb),
				histtype = sty,
				lw = lw,
				color = [(1+ alfa*(_-1)) for _ in yello],
				density = True
				)
		xi = linspace(xmin, xmax, 1001)
		yi = norm.pdf(xi, *calib.T47(T = t))
		plot(xi, yi, 'k-', lw = 1, dashes = (6,1,2,1))
		text(.97, .92, spl[k], va = 'top', ha = 'right', transform = ax.transAxes, weight = 'bold', size = 10)
		text((calib.T47(T = t)[0]-xmin)/(xmax-xmin), .05, f'T = {t:.0f}$\\,$°C', va = 'bottom', ha = 'center', transform = ax.transAxes, weight = 'bold', size = 9)
		xlabel('Δ$_{47}$ (‰)')
		grid(False)
		x0 = round(D47[t].mean() / 0.001) * 0.001
		xticks([x0 - 0.005, x0, x0 + 0.005])
		yticks([])
# 		ax.xaxis.set_major_locator(ticker.MultipleLocator(0.005))
		axis([xmin, xmax, 0, None])


def unical_confidence_limits():

	fig = figure(figsize = (6,4))
	subplots_adjust(.15, .13, .96, .94, .1, .32)
	plot_95CL(sp = (221, 223))
	qmc(Ti = [0., 25, 1000], sp = [322, 324, 326], spl = 'cde')
	savefig('output/fig-16.pdf')
	close(fig)

def export_str(k, v, fid):
	fid.write(f'\n\\define@key{{py}}{{{k}}}[0]{{{v}}}')

def toyexample():
	fig = figure(figsize = (7.2,3.5))
	subplots_adjust(left = 0.07, right = 0.98, bottom = 0.13, top = 0.76, wspace = 0.33)
	
	kw_plot = dict(
		ls = 'None',
		marker = 's',
		mfc = 'w',
		mec = 'k',
		mew = 0.8,
		ms = 5,
		)
        
	kw_true = dict(
		ls = 'None',
		marker = 'o',
		mfc = 'k',
		mec = 'k',
		mew = 0.8,
		ms = 3,
		zorder = 200,
		)
        
	kw_eb = dict(
		ecolor = 'k',
		elinewidth = 0.8,
		capthick = 0.8,
		capsize = 4,
		)

	kw_bff = dict(
		ls = '-',
		marker = 'None',
		color = 'k',
		lw = .8,
		)
	
	kw_ell = dict(
		ec = 'k',
		lw = 0.6,
# 		ls = (0, (4,2)),
		zorder = 100,
		)
	
	
	ax = subplot(131)
	xlabel('$\\mathit{x}$')
	ylabel('$\\mathit{y}$')

	Xtrue = array([10, 20, 40])
	Ytrue = array([20, 30, 50])

	X = array([10, 20, 40])
	Y = array([20, 30, 60])
	sY = array([1, 1, 10])

	xi = array([X[0]-9, X[-1]+9])
	kw_bff['xi'] = xi
	
	WLS = ogls.Polynomial(X = X, Y = Y, sY = sY, degrees = [0,1])
	WLS.regress()

	sY = array([1, 1, 1])
	OLS = ogls.Polynomial(X = X, Y = Y, sY = sY, degrees = [0,1])
	OLS.regress()

	truth = ogls.Polynomial(X = Xtrue, Y = Ytrue, sY = sY, degrees = [0,1])
	truth.regress()
	
	truth.plot_bff(label = f'True ($\\mathit{{y}}$ = $\\mathit{{x}}$ + {truth.bfp["a0"]:.0f})', **{**kw_bff, **dict(lw = 3, color = yello)})
	OLS.plot_bff(dashes = (3,2), label = f'OLS ($\\mathit{{y}}$ = {OLS.bfp["a1"]:.2f} $\\mathit{{x}}$ + {OLS.bfp["a0"]:.1f})', **kw_bff)
	WLS.plot_bff(dashes = (8,2,2,2), label = f'WLS ($\\mathit{{y}}$ = {WLS.bfp["a1"]:.2f} $\\mathit{{x}}$ + {WLS.bfp["a0"]:.1f})', **kw_bff)
	WLS.plot_error_bars(**kw_eb)
	WLS.plot_data(**kw_plot)
	plot(Xtrue, Ytrue, **kw_true)
	plot(X, Y, **kw_plot)
	
	legend(loc = 'lower center', bbox_to_anchor = (0.5, 1.02), fontsize = 8)
	text(.05, .97, 'A', va = 'top', ha = 'left', size = 11, weight = 'bold', transform = ax.transAxes)
	axis([*xi, None, None])
	ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
	
	
	
	ax = subplot(132)
	xlabel('$\\mathit{x}$')
	ylabel('$\\mathit{y}$')

	Xtrue = array([10, 20, 30])
	Ytrue = array([20, 30, 40])

	X = array([10, 20, 28])
	Y = array([20, 30, 42])

	sX = array([1, 1, 1])
	sY = array([1, 1, 1])
	sYX = diag(array([0.9, 0.9, -0.9]))

	xi = array([X[0]-9, X[-1]+9])
	kw_bff['xi'] = xi
	
	ODR = ogls.Polynomial(X = X, Y = Y, sY = sY, sX = sX, degrees = [0,1])
	ODR.regress()

	york = ogls.Polynomial(X = X, Y = Y, sY = sY, sX = sX, sYX = sYX, degrees = [0,1])
	york.regress()
# 	print(york.red_chisq**.5, york.chisq_pvalue)

	truth = ogls.Polynomial(X = Xtrue, Y = Ytrue, sY = sY, sX = sX, sYX = sYX, degrees = [0,1])
	truth.regress()
	
	truth.plot_bff(label = f'True ($\\mathit{{y}}$ = $\\mathit{{x}}$ + {truth.bfp["a0"]:.0f})', **{**kw_bff, **dict(lw = 3, color = yello)})
	ODR.plot_bff(dashes = (3,2), label = f'ODR ($\\mathit{{y}}$ = {ODR.bfp["a1"]:.2f} $\\mathit{{x}}$ + {ODR.bfp["a0"]:.1f})', **kw_bff)
	york.plot_bff(dashes = (8,2,2,2), label = f'York ($\\mathit{{y}}$ = {york.bfp["a1"]:.2f} $\\mathit{{x}}$ + {york.bfp["a0"]:.1f})', **kw_bff)
	york.plot_error_ellipses(p = 0.99, **kw_ell)
	york.plot_data(**kw_plot)
	plot(Xtrue, Ytrue, **kw_true)
	plot(X, Y, **kw_plot)
	
	legend(loc = 'lower center', bbox_to_anchor = (0.5, 1.02), fontsize = 8)
	text(.05, .97, 'B', va = 'top', ha = 'left', size = 11, weight = 'bold', transform = ax.transAxes)
	axis([*xi, None, None])
	ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
	

	
	ax = subplot(133)
	xlabel('$\\mathit{x}$')
	ylabel('$\\mathit{y}$')

	Xtrue = array([10, 20, 30, 40])
	Ytrue = Xtrue + 10

	X = array([9, 19, 31, 41])
	Y = array([21, 31, 39, 49])

	sX = array([[1, .99, 0, 0], [.99, 1, 0, 0], [0, 0, 1, .99], [0, 0, .99, 1]])
	sY = sX
	sYX = array([[-0.9, -0.9, 0, 0], [-0.9, -0.9, 0, 0], [0, 0, -.9, -.9], [0, 0, -.9, -.9]])
	
	xi = array([X[0]-9, X[-1]+9])
	kw_bff['xi'] = xi
	
	OGLS = ogls.Polynomial(X = X, Y = Y, sY = sY, sX = sX, sYX = sYX, degrees = [0,1])
	OGLS.regress()

	york = ogls.Polynomial(X = X, Y = Y, sY = diag(diag(sY)), sX = diag(diag(sX)), sYX = diag(diag(sYX)), degrees = [0,1])
	york.regress()

	truth = ogls.Polynomial(X = Xtrue, Y = Ytrue, sY = sY, sX = sX, sYX = sYX, degrees = [0,1])
	truth.regress()
	
	truth.plot_bff(label = f'True ($\\mathit{{y}}$ = $\\mathit{{x}}$ + {truth.bfp["a0"]:.0f})', **{**kw_bff, **dict(lw = 3, color = yello)})
	york.plot_bff(dashes = (3,2), label = f'York ($\\mathit{{y}}$ = {york.bfp["a1"]:.2f} $\\mathit{{x}}$ + {york.bfp["a0"]:.1f})', **kw_bff)
	OGLS.plot_bff(dashes = (8,2,2,2), label = f'WTLS$\\,/\\,$OGLS ($\\mathit{{y}}$ = {OGLS.bfp["a1"]:.2f} $\\mathit{{x}}$ + {OGLS.bfp["a0"]:.1f})', **kw_bff)
	OGLS.plot_error_ellipses(p = 0.99, **kw_ell)
	OGLS.plot_data(**kw_plot)
	
	legend(loc = 'lower center', bbox_to_anchor = (0.5, 1.02), fontsize = 8)
	text(.05, .97, 'C', va = 'top', ha = 'left', size = 11, weight = 'bold', transform = ax.transAxes)
	axis([*xi, None, None])
	ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
	plot(Xtrue, Ytrue, **kw_true)
	plot(X, Y, **kw_plot)
# 	ax.text(X[0],Y[0]+3,'1')
# 	ax.text(X[1],Y[1]+2,'2')
# 	ax.text(X[2]-2,Y[2]-4,'3')
# 	ax.text(X[3]-2,Y[3]-5,'4')

	savefig('output/fig-02.pdf')
	close(fig)

def poly_degrees_table():

	tex = '\\begin{tabular}{llcrr}\n'
	tex += '\\toprule\n'
	tex += '\\textbf{Degrees} & \\textbf{Model} & \\textbf{RMSWD} & \\textbf{BIC} & \\textbf{AIC} \\\\\n'
	tex += '\\midrule\n'

	for degrees in (
		[0,1],
		[0,2],
		[0,1,2],
		[0,2,3],
		[0,1,2,3],
	# 	[0,1,2,3,4],
		):

		calib = D47calib(
			samples = ogls_2023.samples,
			T = ogls_2023.T,
			sT = ogls_2023.sT,
			D47 = ogls_2023.D47,
			sD47 = ogls_2023.sD47,
			degrees = degrees,
			)

		degstr = '$(' + ','.join([str(d) for d in degrees]) + ')$'
		model = '$\\text{{\\Δ47}} = a_0 + ' + ' + '.join([f'a_{d}/T^{{{d if d>1 else ""}}}' for d in degrees if d]) + '$'
		tex += f'{degstr} & {model} & ${calib.red_chisq**.5:.2f}$ & ${calib.bic:.1f}$ & ${calib.aic:.1f}$ \\\\\n'
# 		print(degrees, calib.chisq_pvalue, calib.red_chisq)

	tex += '\\bottomrule\n'
	tex += '\\end{tabular}\n'

# 	with open('../../latex/degrees.tex', 'w') as fid:
	with open('output/degrees.tex', 'w') as fid:
		fid.write(tex)

def wavg():

	from matplotlib.patches import Ellipse

	X = array([-2, 2])
	Y = array([0, 0])
	sX = array([[1, 0], [0, 1]])
	sY = array([[1, 0], [0, 1]])
	N = X.size

	fig = figure(figsize = (3.5, 3.625))
	subplots_adjust(
		left = .15, right = .95, # 0.8 * 3.5 = 2.8
		bottom = .12, top = 0.97, # 0.7 * 2 = 1.4
		hspace = .2,
		)

	for _, lbl, sYX in [
		(211, 'A', array([[0, 1], [-1, 0]]) * 0.9),
		(212, 'B', array([[0, 1], [1, 0]]) * 0.9),
		]:
	
		ax = subplot(_)
		Z = array([X, Y])
		sZ = block([[sX, sYX.T], [sYX, sY]])

		W = ogls.WeightedMean(Z, sZ)
		W.regress()

		x, y = W.bfp['X0'], W.bfp['X1']
		CM = W.bfp_CM


		for k in range(Z.shape[1]):
			w,h,r = ogls.cov_ellipse(array([[sZ[k,k], sZ[k,k+N]], [sZ[k+N,k], sZ[k+N,k+N]]]))
			ax.add_patch(
				Ellipse(
					xy = (X[k], Y[k]),
					width = w,
					height = h,
					angle = r,
					fc = 'w',
					ec = 'k',
					lw = 1,
					alpha = 0.5,
					)
				)

		w,h,r = ogls.cov_ellipse(CM)

		ax.add_patch(
			Ellipse(
				xy = (x, y),
				width = w,
				height = h,
				angle = r,
				fc = yello,
				ec = 'k',
				lw = 1,
				alpha = 0.5,
				)
			)

		text(
			0.02, 0.04,
			lbl,
			va = 'bottom', ha = 'left',
			size = 9,
			weight = 'bold',
			transform = ax.transAxes,
			)

		text(
			0.02, 0.96,
			f'$\\mathit{{ρ_{{x_1y_2}}}} = {sYX[1,0]}$\n$\\mathit{{ρ_{{x_2y_1}}}} = {sYX[0,1]}$',
			va = 'top', ha = 'left',
			size = 9,
			transform = ax.transAxes,
			)

		plot(x, y, 's', mec = 'k', mew = 1, mfc = yello)
		plot(X, Y, 's', mec = 'k', mew = 1, mfc = 'w')
		for k, (x,y) in enumerate(zip(X,Y)):
			text(x, y, f'{k+1}\n', va = 'center', ha = 'center', linespacing = 2.5)
	
		axis(array([-2.3, 1.7, -1, 1])*3)
		if _ == 212:
			xlabel('$\\mathit{x}$')
		ylabel('$\\mathit{y}$', rotation = 0, labelpad = 10)

	savefig('output/fig-08.pdf')
	close(fig)


if __name__ == '__main__':

	shutil.rmtree('output', ignore_errors = True)
	Path('output').mkdir(exist_ok=True)
	

	with open('output/py.tex', 'w') as libexport:
		libexport.write(r'''
\usepackage{xkeyval}
\makeatletter
			'''[1:-3])

		print('Weighted average')
		wavg()

		print('Polynomial degrees table')
		poly_degrees_table()
	
		print('Toy example plots')
		toyexample()

		print('Pearson-York plot')
		fig_pearson_york()

		print('sYX example')
		fig_example_sYX()

		print('sY example')
		fig_example_sY()

		print('Convergence issue plot')
		fig_convergence_issue()

		print('D47correl plot')
		D47correl(anderson_2021_mit, 'D47correl_A21_MIT', -.1, 0.4)
		D47correl(anderson_2021_lsce, 'D47correl_A21_LSCE', -.1, 0.4)
		D47correl(breitenbach_2018, 'D47correl_B18', -.1, 0.4)
		D47correl(peral_2018, 'D47correl_P18', -.1, 0.4)
		D47correl(fiebig_2021, 'D47correl_F21', 0, 0.5)
		D47correl(jautzy_2020, 'fig-10a', 0, 0.5)
		D47correl(huyghe_2022, 'D47correl_H22', -.1, 0.4)
		D47correl(devils_laghetto_2023, 'D47correl_D23', -.1, 0.4)

		print('Tcorrel plot')
		Tcorrel(peral_2018, 'fig-10b', 0.85, 1, reorder = False)
		Tcorrel(devils_laghetto_2023, 'Tcorrel_DL23', 0.85, 1, reorder = False)

		print('Unical plot')
		unicalib_plot()

		print('Unical confidence limits')
		unical_confidence_limits()

		print('KS plots')
		KS_plots()

		print('Calibration plots')
		calib_plots()

		print('Compares data sets')
		conf_limits()
		conf_ellipses()

		print('Compare with theory')
		compare_with_theory()

		print('Equation rounding')
		unicalib_rounding()
			
		libexport.write(r'''
\makeatother
\newcommand{\py}[1]{\setkeys{py}{#1}}''')
