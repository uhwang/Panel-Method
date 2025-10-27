'''
    Source Panel Method 2025
    Uisang Hwang
    10/25/2025
    python setup.py build_ext --inplace
'''
import numpy as np
import libvgl as vgl
import naca45 as nc
import streamlines as stl

fps, dur = 5, 30
min_aoa, max_aoa, naoa = 0., 6, fps*dur
daoa = (max_aoa-min_aoa)/naoa

class SPMGeom:
    def __init__(self, U, alpha, sx, sy, npan, radius):
        self.U  = U
        self.alpha = np.deg2rad(alpha)
        self.sx = sx
        self.sy = sy
        self.npan = npan
        self.radius = radius
        self.X = None
        self.Y = None
        # Control Point 
        self.CX    = np.zeros(npan, dtype=np.float64)
        self.CY    = np.zeros(npan, dtype=np.float64)
        # Normal Vector
        self.NX    = np.zeros(npan, dtype=np.float64)
        self.NY    = np.zeros(npan, dtype=np.float64)
        # Angle from x-axis to panel in CCW
        self.phi   = np.zeros(npan, dtype=np.float64)
        # Angle from x-axis to normal vector in CCW
        self.delta = np.zeros(npan, dtype=np.float64)
        # Angle between normal vector and U
        self.beta  = np.zeros(npan, dtype=np.float64)
        # Panel Length
        self.SJ    = np.zeros(npan, dtype=np.float64)
        # Matrix A
        self.A     = np.zeros((npan,npan), dtype=np.float64)
        # Matrix B
        self.B     = np.zeros(npan, dtype=np.float64)
        # Source strength
        self.sigma = np.zeros(npan, dtype=np.float64)
        # Pressucre Coefficient
        self.CP    = np.zeros(npan, dtype=np.float64)
        # Tangential Velocity
        self.VT    = np.zeros(npan, dtype=np.float64)

def create_geom(geom):
    g = geom
    for i in range(g.npan):
        x1 = g.X[i]
        y1 = g.Y[i]
        x2 = g.X[i+1]
        y2 = g.Y[i+1]
        x21 = x2-x1
        y21 = y2-y1
        g.SJ[i] = np.sqrt(x21**2+y21**2)
        g.CX[i] = x1 + x21*0.5
        g.CY[i] = y1 + y21*0.5
        phi = np.atan2(y21,x21)
        if phi < 0:
            phi += np.pi*2
        g.phi  [i] = phi
        g.delta[i] = phi+np.pi*0.5
        nv         = [-y21, x21] 
        g.NX[i]    = nv[0]/g.SJ[i]
        g.NY[i]    = nv[1]/g.SJ[i]
    
    g.delta[g.delta>2*np.pi] = g.delta[g.delta > 2*np.pi] - 2*np.pi
    g.beta = g.delta - g.alpha
   
def create_geom_circle(geom):
    g = geom
    deg_shift_amount = 360/g.npan*0.5
    vg = vgl.Polygon(g.sx, g.sy, g.npan, g.radius, 
                    deg_shift=deg_shift_amount, 
                    end_point=True, ccw=False)
    g.X = vg.xss.astype(np.float64)
    g.Y = vg.yss.astype(np.float64)
    create_geom(g)
    
def create_geom_foil(geom, fcode):
    g = geom
    xss, yss = nc.get_naca45(fcode, g.npan, nc.spc_cos)
    g.X = xss.astype(np.float64)
    g.Y = yss.astype(np.float64)
    create_geom(g)
    
def BIJ(geom):
    g = geom
    
    for i in range(g.npan):
        vti = 0
        for j in range(g.npan):
            if j == i: continue
            A = -(g.CX[i]-g.X[j])*np.cos(g.phi[j]) \
                -(g.CY[i]-g.Y[j])*np.sin(g.phi[j])  
            B = (g.CX[i]-g.X[j])**2+(g.CY[i]-g.Y[j])**2        
            C = -np.cos(g.phi[j]-g.phi[i])
            D = (g.CX[i]-g.X[j])*np.cos(g.phi[i]) \
               +(g.CY[i]-g.Y[j])*np.sin(g.phi[i])  
            E = np.sqrt(B-A**2)
            P = g.SJ[j]**2+2*A*g.SJ[j]+B   
            k1 = C/2*np.log(P/B)
            k2 = (D-C*A)/E*(np.atan2((g.SJ[j]+A),E)- \
                            np.atan2(A,E))
            vti += g.sigma[j]/(np.pi*2)*(k1+k2)
        g.VT[i] = g.U*np.sin(g.beta[i])+vti
            
def AIJ(geom):
    g = geom
    for i in range(g.npan):
        for j in range(g.npan):
            if j == i:
                g.A[i,j] = 1/2
            else:
                A = -(g.CX[i]-g.X[j])*np.cos(g.phi[j]) \
                    -(g.CY[i]-g.Y[j])*np.sin(g.phi[j])  
                B = (g.CX[i]-g.X[j])**2+(g.CY[i]-g.Y[j])**2
                C = np.sin(g.phi[i]-g.phi[j])
                D = (g.X[j]-g.CX[i])*np.sin(g.phi[i]) \
                   -(g.Y[j]-g.CY[i])*np.cos(g.phi[i])  
                E = np.sqrt(B-A**2)
                P = g.SJ[j]**2+2*A*g.SJ[j]+B
                k1 = C/2*np.log(P/B)
                k2 = (D-C*A)/E*(np.atan2((g.SJ[j]+A),E)- \
                                np.atan2(A,E))
                g.A[i,j] = 1/(2*np.pi)*(k1+k2)
        g.B[i] = -g.U*np.cos(g.beta[i]) 

def solve(geom):
    #print("... Create Influence Coeff Matrix")
    AIJ(geom)
    #print("... Solve Linear Equation")
    geom.sigma = np.linalg.solve(geom.A, geom.B)
    #print("... Create Tangential Velocity Matrix")
    BIJ(geom)
    #print("... Compute CP")
    geom.CP = 1-(geom.VT/geom.U)**2
    #print(geom.CP)
    
def plot_geom(dev, frm, geom):
    g = geom
    dev.set_device(frm)
    
    dev.polyline(g.X, g.Y)
    dev.line(g.X[0], g.Y[0], g.X[1], g.Y[1], lcol=vgl.GREEN, lthk=0.005)
    # plot normal vector
    for i, (cx,cy) in enumerate(zip(g.CX,g.CY)):
         dev.arrow(cx, cy, 
                   cx+g.NX[i]*0.3, 
                   cy+g.NY[i]*0.3, "-|>lc(b)lt(0.003)")

    # plot symbol at control point
    vgl.plot_circle_symbol(dev, g.X, g.Y)
    vgl.plot_circle_symbol(dev, g.CX, g.CY, fcol=vgl.GREEN)
    vgl.draw_axis(dev)

    # plot Delta (apgle from x-axis to normal vector)
    for i in range(g.npan):
        # plot x-axis
        dev.line(g.CX[i], g.CY[i], 
                 g.CX[i]+g.SJ[0]*0.6, g.CY[i], 
                 lthk=0.002, lpat=vgl.get_dash(0.007))
        # plot Phi : from x-axis to panel
        vgl.arc(dev, g.CX[i], g.CY[i], 
                     g.SJ[0]*0.10, 0, 
                     vgl.rad_to_deg(g.phi[i]), lcol_out=vgl.RED)
        # plot delta : from x-axis to normal vector
        vgl.arc(dev, g.CX[i], g.CY[i], 
                     g.SJ[0]*0.20, 0, 
                     vgl.rad_to_deg(g.delta[i]), lcol_out=vgl.BLUE)
        
        # plot U vector and AOA: alpha
        dev.arrow(g.CX[i], g.CY[i], 
                  g.CX[i]+0.3*np.cos(g.alpha),
                  g.CY[i]+0.3*np.sin(g.alpha),
                  "-|>lc(b)lt(0.003)")
        # plot Beta from U to Normal vector
        vgl.arc(dev, g.CX[i], g.CY[i], 
                     g.SJ[0]*0.3, 
                     vgl.r2d(g.delta[i]), 
                     vgl.r2d(g.delta[i]+np.pi*2-g.beta[i]))

def plot_streamlines(dev, frm, geom, xmin, xmax, ymin, ymax, nx, ny):
    g = geom
    dev.set_device(frm)
    lines = stl.get_streamlines_rk4(geom, xmin, xmax, ymin, ymax, nx, ny)
    vgl.print_top_center(dev, f"\\alpha={np.rad2deg(g.alpha):2.2f}\\deg")

    if lines == []: return

    #dev.polyline(g.X, g.Y, lcol=vgl.BLACK, lthk=0.001)
    dev.polygon(g.X, g.Y, lcol=vgl.BLACK, lthk=0.001, fcol=vgl.YELLOW)
    #vgl.plot_circle_symbol(dev, g.X, g.Y, size=0.009, lcol=vgl.RED)
    #vgl.plot_circle_symbol(dev, g.CX, g.CY, size=0.009, lcol=vgl.PURPLE, fcol=vgl.PURPLE)
    
    for line in lines:
        dev.polyline(line[:,0], line[:,1], vgl.color.BLUE)
    vgl.draw_axis(dev)    

def run(dev):
    global frm1, frm2
    global sminx, smaxx, sminy, smaxy
    print("... Create Geometry")
    create_geom_circle(spm_geom)
    #create_geom_foil(spm_geom)
    solve(spm_geom)
    plot_geom(dev, frm1, spm_geom)
    plot_streamlines(dev, frm2, spm_geom, sminx, smaxx, sminy, smaxy, 30, 30)
    dev.close()
    
def run_spm(t):
    global sminx, smaxx, sminy, smaxy
    global dev, frm1, frm2
    npan, radius = 40, 0.5

    aoa_d = min_aoa+t*daoa*fps
    spm_circ = SPMGeom(1,aoa_d, radius, 0, npan*2, radius)
    spm_foil = SPMGeom(1,aoa_d, radius, 0, 200, radius)
    create_geom_foil(spm_foil, "4412")
    create_geom_circle(spm_circ)
    
    #solve(spm_circ)
    solve(spm_foil)
    
    dev.fill_white()
    #plot_streamlines(dev, frm1, spm_circ, sminx, smaxx, sminy, smaxy, 30, 30)
    plot_streamlines(dev, frm2, spm_foil, sminx, smaxx, sminy, smaxy, 30, 30)
        
def save():
    global frm1, frm2, frm3, dev
    global sminx, smaxx, sminy, smaxy# = -4, 4, -4, 4
    sminx, smaxx, sminy, smaxy = -0.5, 1.5, -1, 1
    fmm = vgl.FrameManager()
    #frm1 = fmm.create(0, 0, 4, 4, vgl.Data(sminx, smaxx, sminy, smaxy))
    frm2 = fmm.create(4, 0, 4, 4, vgl.Data(sminx, smaxx, sminy, smaxy))
    dev = vgl.DeviceIMG("", fmm.get_gbbox(), 300)
    
    dev_mov = vgl.DeviceCairoAnimation("spm.mp4", dev, run_spm, dur, fps)
    dev_mov.save_video()

    #dev_ppt = vgl.DevicePPT("spm_circle.pptx", fmm.get_gbbox())
    #dev_img = vgl.DeviceIMG("spm_circle.jpg", fmm.get_gbbox(), 400)
    
    #run(dev_ppt,frm)
    
if __name__ == "__main__":
    save()