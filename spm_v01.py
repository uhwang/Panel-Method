'''
    Source Panel Method 2025
    Uisang Hwang
    10/25/2025
'''
import numpy as np
import libvgl as vgl
import naca45 as nc
# --- Define a global or constant time step for RK4 ---
# Choose a time step that's small relative to the flow speed and panel size.
# Since U=1 and your geometry is small (radius=0.5), DT=0.05 is a decent start.
DT = 0.05 

def get_streamlines_rk4(geom, 
                        Vxyp, 
                        minx, maxx, miny, maxy,
                        nx, ny, dt=DT): # dt is the time step
    
    g = geom
    
    # Define start lines based on gridy (just like your original code)
    gridy = np.linspace(miny, maxy, ny, endpoint=True)
    startx = minx

    stream_lines = list()
    
    for yval in gridy:
        x1 = startx
        y1 = yval
        
        sub_lines = list()
        sub_lines.append((x1, y1))
        
        # --- RK4 Integration Loop ---
        while True:
            # 1. Check if point is outside domain
            if (x1 < minx or x1 > maxx or 
                y1 < miny or y1 > maxy):
                break
                
            # --- RK4 Steps ---
            # K1: Velocity at (x1, y1)
            u1, v1 = Vxyp(g, x1, y1)
            
            # K2: Velocity at midpoint using K1
            x_k2 = x1 + u1 * dt / 2
            y_k2 = y1 + v1 * dt / 2
            u2, v2 = Vxyp(g, x_k2, y_k2)
            
            # K3: Velocity at midpoint using K2
            x_k3 = x1 + u2 * dt / 2
            y_k3 = y1 + v2 * dt / 2
            u3, v3 = Vxyp(g, x_k3, y_k3)
            
            # K4: Velocity at end point using K3
            x_k4 = x1 + u3 * dt
            y_k4 = y1 + v3 * dt
            u4, v4 = Vxyp(g, x_k4, y_k4)
            
            # --- RK4 Averaging and Update ---
            
            # Average the slopes (velocities)
            avg_u = (u1 + 2*u2 + 2*u3 + u4) / 6
            avg_v = (v1 + 2*v2 + 2*v3 + v4) / 6
            
            # Calculate new position
            x_new = x1 + avg_u * dt
            y_new = y1 + avg_v * dt

            # Check for stagnation point or extremely small velocity
            # Prevents infinite loop if V ~ 0 (e.g., at stagnation point)
            # We use a tolerance (TOL) based on freestream speed.
            TOL = 1.0E-6 * g.U 
            
            if np.sqrt(avg_u**2 + avg_v**2) < TOL:
                break 

            # Update and continue tracing
            sub_lines.append((x_new, y_new))
            x1 = x_new
            y1 = y_new
        
        stream_lines.append(sub_lines)
            
    return stream_lines
    
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
        self.CX    = np.zeros(npan)
        self.CY    = np.zeros(npan)
        # Normal Vector
        self.NX    = np.zeros(npan)
        self.NY    = np.zeros(npan)
        # Angle from x-axis to panel in CCW
        self.phi   = np.zeros(npan)
        # Angle from x-axis to normal vector in CCW
        self.delta = np.zeros(npan)
        # Angle between normal vector and U
        self.beta  = np.zeros(npan)
        # Panel Length
        self.SJ    = np.zeros(npan)
        # Matrix A
        self.A     = np.zeros((npan,npan))
        # Matrix B
        self.B     = np.zeros(npan)
        # Source strength
        self.sigma = np.zeros(npan)
        # Pressucre Coefficient
        self.CP    = np.zeros(npan)
        # Tangential Velocity
        self.VT    = np.zeros(npan)


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
    g.X, g.Y = vg.xss, vg.yss
    create_geom(g)
    
def create_geom_foil(geom, fcode):
    g = geom
    g.X, g.Y = nc.get_naca45(fcode, g.npan, nc.spc_cos)
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

def Vxy(geom, xp, yp):
    g = geom
    kx, ky = 0,0
    
    for j in range(g.npan):
        A  = -(xp-g.X[j])*np.cos(g.phi[j]) \
             -(yp-g.Y[j])*np.sin(g.phi[j])
        B  = (xp-g.X[j])**2+(yp-g.Y[j])**2
        E = np.sqrt(B-A**2)
        P = g.SJ[j]**2+2*A*g.SJ[j]+B
        
        Cx = -np.cos(g.phi[j])
        Dx = xp - g.X[j]
        
        Cy = -np.sin(g.phi[j])
        Dy = yp - g.Y[j]
        
        kx1 = (Cx/2*np.log(P/B))
        kx2 = (Dx-Cx*A)/E*(np.atan((g.SJ[j]+A)/E)- \
                            np.atan(A/E))
        
        ky1 = (Cy/2*np.log(P/B))
        ky2 = (Dy-Cy*A)/E*(np.atan((g.SJ[j]+A)/E)- \
                            np.atan(A/E))
        kx += g.sigma[j]/(np.pi*2)*(kx1 + kx2)
        ky += g.sigma[j]/(np.pi*2)*(ky1 + ky2)
    vx = g.U*np.cos(g.alpha) + kx
    vy = g.U*np.sin(g.alpha) + ky
    
    return vx, vy
    
def solve(geom):
    print("... Create Influence Coeff Matrix")
    AIJ(geom)
    print("... Solve Linear Equation")
    geom.sigma = np.linalg.solve(geom.A, geom.B)
    print("... Create Tangential Velocity Matrix")
    BIJ(geom)
    print("... Compute CP")
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
    lines = get_streamlines_rk4(geom, Vxy, xmin, xmax, ymin, ymax, nx, ny)
    
    if lines == []: return

    #dev.polyline(g.X, g.Y, lcol=vgl.BLACK, lthk=0.001)
    dev.polygon(g.X, g.Y, lcol=vgl.BLACK, lthk=0.001, fcol=vgl.YELLOW)
    vgl.plot_circle_symbol(dev, g.X, g.Y, size=0.004, lcol=vgl.RED, fcol=vgl.RED)
    #vgl.plot_circle_symbol(dev, g.CX, g.CY, size=0.01, lcol=vgl.PURPLE, fcol=vgl.PURPLE)

    for line in lines:
        xx = [l[0] for l in line]
        yy = [l[1] for l in line]   
        dev.polyline(xx, yy, vgl.color.BLUE)
    vgl.draw_axis(dev)    

def run(dev):
    global frm1, frm2
    global sminx, smaxx, sminy, smaxy
    print("... Create Geometry")
    npan, radius = 40, 0.5
    spm_geom = SPMGeom(1,0,radius,0,npan, radius)
    #create_geom_circle(spm_geom)
    create_geom_foil(spm_geom, "4412")
    solve(spm_geom)
    #print(spm_geom.sigma)
    plot_geom(dev, frm1, spm_geom)
    plot_streamlines(dev, frm2, spm_geom, sminx, smaxx, sminy, smaxy, 30, 30)
    dev.close()
    
def save():
    global frm1, frm2
    global sminx, smaxx, sminy, smaxy# = -4, 4, -4, 4
    sminx, smaxx, sminy, smaxy = -0.5, 1.5, -1, 1
    fmm = vgl.FrameManager()
    #frm1 = fmm.create(0, 0, 4, 4, vgl.Data(-0.6, 1.6, -1.1, 1.1))
    frm1 = fmm.create(0, 0, 4, 4, vgl.Data(-0.6, 1.6, -1.1, 1.1))
    frm2 = fmm.create(4, 0, 4, 4, vgl.Data(sminx, smaxx, sminy, smaxy))
    
    dev_ppt = vgl.DevicePPT("spm_circle.pptx", fmm.get_gbbox())
    dev_img = vgl.DeviceIMG("spm_circle.jpg", fmm.get_gbbox(), 400)
    
    #run(dev_ppt,frm)
    run(dev_img)
    
if __name__ == "__main__":
    save()