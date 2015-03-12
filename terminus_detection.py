import psycopg2

import xlrd
import re
import unicodedata
import numpy as N
import datetime as dtm
import os
import glob
import simplekml as kml
import subprocess
from types import *
from osgeo.gdalnumeric import *
from osgeo import gdal
import sys
import matplotlib.pyplot as plt
import time
import copy as c
import scipy.stats.mstats as mstats
from scipy import ndimage
from scipy import misc
import scipy.stats as stats
import cv2 as cv2

sys.path.append('/Users/igswahwsmcevan/Altimetry/code')
#from .Interface import *
from Altimetry.Interface import *
#import Altimetry.descriptive

import ConfigParser
cfg = ConfigParser.ConfigParser()
cfg.read(os.path.dirname(__file__)+'/setup.cfg')

import ppygis as gis
import StringIO
import psycopg2
import ppygis

import matplotlib as mpl
import matplotlib.cm as cm
import random


def moment(alpha, p=1, cent=False,
           w=None, d=None, axis=None,
           ci=None, bootstrap_iter=None):
    """
    Computes the complex p-th centred or non-centred moment of the angular
    data in alpha.

    :param alpha: sample of angles in radian
    :param p:     the p-th moment to be computed; default is 1.
    :param cent:  if True, compute central moments. Default False.
    :param w:     number of incidences in case of binned angle data
    :param d:     spacing of bin centers for binned data, if supplied
                  correction factor is used to correct for bias in
                  estimation of r
    :param axis:  compute along this dimension,
                  default is None (across all dimensions)
    :param ci: if not None, confidence level is bootstrapped
    :param bootstrap_iter: number of bootstrap iterations
                           (number of samples if None)
    :return:    the complex p-th moment.
                rho_p   magnitude of the p-th moment
                mu_p    angle of the p-th moment

    Example:

        import numpy as np
        import pycircstat as circ
        data = 2*N.pi*N.random.rand(10)
        mp = circ.moment(data)

    You can then calculate the magnitude and angle of the p-th moment:

        rho_p = N.abs(mp)  # magnitude
        mu_p = N.angle(mp)  # angle

    You can also calculate bootstrap confidence intervals:

        mp, (ci_l, ci_u) = circ.moment(data, ci=0.95)

    References: [Fisher1995]_ p. 33/34
    """

    if w is None:
        w = N.ones_like(alpha)

    assert w.shape == alpha.shape, "Dimensions of alpha and w must match"

    if cent:
        theta = mean(alpha, w=w, d=d, axis=axis)
        theta2 = N.tile(theta, (alpha.shape[0],) + len(theta.shape) * (1,))
        alpha = cdiff(alpha, theta2)

    n = alpha.shape[axis]
    cbar = N.sum(N.cos(p * alpha) * w, axis) / n
    sbar = N.sum(N.sin(p * alpha) * w, axis) / n
    mp = cbar + 1j * sbar

    return mp
        
    
def import_raster(filepath,dtmparse = '(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})',cameraid=1,locationid=1):
    
    filename = os.path.basename(filepath)
 
    
    datetxt = N.array(re.findall(dtmparse,filename)[0]).astype(int)
    date = dtm.datetime(datetxt[0],datetxt[1],datetxt[2],datetxt[3],datetxt[4],datetxt[5])
    
    # Connect to an existing spatially enabled database
    conn, cur = ConnectDb()
    
    cur.execute("SELECT photoid FROM timelapse WHERE path='%s';" % filepath)
    if len(cur.fetchall())!=0:
        print "WARNING: This image has already been imported, import canceled"
        return None
    
    #IMPORTING RASTER FILE
    os.system("/Applications/Postgres.app/Contents/Versions/9.3/bin/raster2pgsql -a -R -f image -Y %s public.timelapse | /Applications/Postgres.app/Contents/Versions/9.3/bin/psql -d altimetry" % filepath)
    
    #FINDING ID OF LAST ENTRY IN TABLE
    cur.execute("SELECT photoid::int from timelapse order by photoid desc limit 1;")
    photoid = cur.fetchall()[0][0]
    
    #UPDATING OTHER IMAGE METADATA
    cur.execute("UPDATE timelapse SET (cameraid,locationid,date,path) = (%i,%i,'%s','%s') WHERE photoid=%i;" % (cameraid,locationid,re.sub('T',' ', date.isoformat()),filepath,photoid))
    conn.commit()
    
    #FINDING ID OF THIS IMAGE
    cur.execute("SELECT photoid::int from timelapse order by photoid desc limit 1;")
    
    out = cur.fetchall()[0][0]
    cur.close()
    return out
    

    
def import_timelapse_dir(folder, select = "*"):
  
    #getting that backslash in there
    if re.search("[^/]$",folder):folder="%s/" % folder
    
    
    filepaths = glob.glob(folder+select)
    
    for f in filepaths:
        out = import_raster(f)

#class TLimageObject:
#    def __init__(self, photoid=None,imgpath=None):
#        
#        # Connect to an existing spatially enabled database
#        conn, cur = ConnectDb()
#        
#        
#        if 'gid' in indata.keys():self.gid = indata['gid']
#            if 'glid' in indata.keys():self.glid = indata['glid']
            
class TimeImage:
    def __init__(self, thisphoto):
        if type(thisphoto)==str:
            if os.path.isabs(thisphoto):
                self.imagepath = thisphoto
                self.photoid = GetSqlData2("SELECT photoid::int as photoid from timelapse where path = '%s';" % self.imagepath)['photoid'][0]
        elif isinstance(thisphoto, (int, long, float)):
            if GetSqlData2("SELECT EXISTS(SELECT 1 FROM timelapse WHERE photoid=%i)" % thisphoto):
                self.photoid  = thisphoto
                self.imagepath = GetSqlData2("SELECT path FROM timelapse WHERE photoid = '%s';" % self.photoid)['path'][0]
            else: raise "This photoid or path cannot be found"
        else:raise "Please provide a valid photoid OR image path"
        self.img = None
        self.edges = Edges(where='photoid=%i' % self.photoid)
        
    def get_image(self):
        if type(self.img)==NoneType:
            self.img = cv2.imread(self.imagepath,cv2.IMREAD_COLOR)
            return self.img
        else:
            return self.img
    def Canny1(self, threshold1=0,threshold2=200,overwriteallphoto = True,overwriteparamrun = False,binarycutoff = 127,apertureSize=3):

        #CALCULATING EDGES
        print threshold1,threshold2
        edges = cv2.Canny(self.get_image(),threshold1,threshold2,apertureSize=apertureSize)
        ret,thresh = cv2.threshold(edges,binarycutoff,255,0)  
        contours,hierarchy  = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        
        #self.get_lines(asxy=True)
        #self.get_image()
        #cv2.drawContours(self.get_image(), contours, -1, (0,255,0), 2) 
        #cv2.imwrite('/Users/igswahwsmcevan/Desktop/test2.jpg', self.get_image())

        #print type(contours),type(contours[4]),contours[4].shape
        #print type(contours),type(contours[1]),contours[1].shape
        #print contours[4]
        #PREPING THEM TO BE INSERTED AS A GEOMETRY
        start_time = time.time()
        lines=[ppygis.LineString([gis.Point(o[0][0],-o[0][1]) for o in c]) for c in contours]

        #Prepare data for upload to database
        #parameters used for this run
        params = "Canny1 threshold1:%s threshold2:%s binarycutoff:%s " % (threshold1,threshold2,binarycutoff)
        now = re.sub('T',' ',dtm.datetime.now().isoformat())

        #CHECKING REQUESTS FOR OVERWRITES
        conn, cur = ConnectDb()
        if overwriteallphoto:
            cur.execute("DELETE from edgeparameters WHERE photoid=%s;" % self.photoid)
            conn.commit()
            
        if overwriteparamrun:
            cur.execute("DELETE from edgeparameters WHERE photoid=%s AND params='%s';" % (self.photoid,params))
            conn.commit()
        
        
        #INSERTING INTO EDGE PARAMETER TABLE
        cur.execute("INSERT INTO edgeparameters (photoid,params,rundate) VALUES (%i,'%s','%s') RETURNING paramid;" % (self.photoid,params,now))
        paramid = cur.fetchall()[0][0]
        for h,l in zip(hierarchy[0],lines):
            #INSERTING INTO EDGES TABLE 
            cur.execute("INSERT INTO edges (photoid,paramid,presamelvl,nextsamelvl,child,parent,geometry) VALUES (%i,%i,%i,%i,%i,%i,'%s');" % (self.photoid,paramid,h[0],h[1],h[2],h[3],l.write_ewkb()))
        conn.commit()
        cur.close()
        
        #############
        #RETURN AN OBJECT that points to this data in the db AND MAKE THIS A METHOD OF A TIMELAPSE OBJECT
        self.edges = Edges(where="photoid = %s" % self.photoid,presamelvl=hierarchy.T[0],nextsamelvl=hierarchy.T[1],child=hierarchy.T[2],parent=hierarchy.T[3],lines=lines,paramid=paramid)

        return self.edges  
        
    def get_lines(self, getfields = None, where=None,asxy=False, negy = True, simplified=None):
        
              #THIS METHOD WILL SELECT THE APPROPRIATE EDGES FROM THE DATABASE, UNLESS THEY ARE ALREADY LOADED INTO MEMORY
              #IN WHICH CASE IT JUST RETURNS THE EDGES FROM MEMORY.  YOU CAN ALSO SPECIFY A SPECIFIC WHERE FOR A QUERY      
        if type(where)==NoneType and not type(self.edges.lines)==NoneType:
            if asxy:
                if negy: 
                    return [N.array([[pt.x,pt.y] for pt in ls.points]) for ls in self.edges.lines]
                else:
                    return [N.array([[pt.x,-pt.y] for pt in ls.points]) for ls in self.edges.lines]
            else:
                return self.edges

        if type(where) == NoneType:
            toggle = True
            where='WHERE photoid = %i' % self.photoid
        else: 
            if not re.search('^\s*WHERE',where):
                where = "WHERE %s" % where

            if not re.search('photoid\s*=\s*%i' % self.photoid,where):
                if re.search('photoid\s*=',where):raise "ERROR: here you can only query edges from this photo"
                where= "%s AND photoid=%i" % (where,self.photoid)

            toggle = False
            
        #start_time = time.time()
        conn = psycopg2.connect(database = 'altimetry', host='localhost')
        cur = conn.cursor()
        
        if type(getfields)==NoneType:fields = 'geometry'
        else: 
            if re.search('geometry',getfields):fields=getfields
            else:fields = "%s ,geometry" % getfields

        #QUERYING DATABASE
        print "SELECT %s FROM edges %s;" % (fields,where)
        cur.execute("SELECT %s FROM edges %s;" % (fields,where))
        
        #getting fields 
        fields = [column[0] for column in cur.description]
        
        #this takes the database output puts it in columns then into a dictionary with field names as keys
        out = dict(zip(fields,zip(*cur.fetchall())))
        cur.close()
        

        out['lines'] = [ppygis.Geometry.read_ewkb(row) for row in out['geometry']]
        out.pop('geometry')
        out['where']=where
        
        #option to output the coordinates just as an xy list not as ppygis objects
        if asxy: 
            if negy:
                return [N.array([[pt.x,pt.y] for pt in ls.points]) for ls in out['lines']]
            else:
                return [N.array([[pt.x,-pt.y] for pt in ls.points]) for ls in out['lines']]

            
        if toggle and not asxy:
            self.edges = Edges(**out)
            return self.edges
        else:
            return Edges(**out)
         
    def saveimg(self, outpath = None, showedges = True,setlims=None,coloredgesby=None):
        import matplotlib.collections as col
        
        fig = plt.figure(figsize=[self.get_image().shape[1]/300.,self.get_image().shape[0]/300.],dpi = 300)
        ax = fig.add_axes([0,0,1,1],frameon=False)
        ax.imshow(self.get_image()[:,:,::-1])
        
        if showedges:
            if type(coloredgesby)!=NoneType:
                
                if coloredgesby == 'random':
                    colorby = N.random.rand(len(a.get_lines().lines))
                else:
                    colorby = self.edges.get_attribute('straightness1')

        
                norm = mpl.colors.Normalize(vmin=colorby.min(), vmax=colorby.max())
                cmap = cm.hot
        
                
                m = cm.ScalarMappable(norm=norm, cmap=cmap)
                colors=m.to_rgba(colorby)  
        
            self.get_lines()
            ln_coll = col.LineCollection(self.get_lines(asxy=True, negy=False),lw=0.3,color=colors)
            ax.add_collection(ln_coll)
            
        if type(setlims)!=NoneType:
            ax.set_xlim(setlims[0],setlims[0]+setlims[2])
            ax.set_ylim(setlims[1],setlims[1]-setlims[3])
            
        plt.draw()
        if type(outpath)==str:
            fig.savefig(outpath,dip=300)
            
        else:
            plt.show()



class Edges:
    def __init__(self,where = None,lines = None,edgeid=None,photoid =None,presamelvl=None,nextsamelvl=None,child=None,parent=None,selected=None,front=None,paramid=None):

        self.where = where
        self.lines = lines
        self.edgeid=edgeid
        self.photoid =photoid
        self.presamelvl=presamelvl
        self.nextsamelvl=nextsamelvl
        self.child=child
        self.parent=parent
        self.selected=selected
        self.front=front
        self.paramid=paramid
        self.straightness1=None
        
    def retrieve_fields(self,getfields):
        if type(getfields)==str:fields = re.split('\s*,?\s*',getfields)
        elif type(getfields)==list:fields=getfields
        elif type(getfields)==NoneType:fields = 'geometry'
        else:raise 
        
        if self.where==None:raise "EdgeObject needs a where attribute to run get_fields"
        
        #print 'where',self.where
        for i,field in enumerate(fields):
            if field == 'edgeid':fields[i]='edgeid::int as edgeid'
            if field == 'geometry':field='lines'
        
        #start_time = time.time()
        conn = psycopg2.connect(database = 'altimetry', host='localhost')
        cur = conn.cursor()
        
        #print ','.join(fields)
        if not re.search('^\s*WHERE',self.where):where = "WHERE %s" % self.where
        else:where=self.where
        
        #QUERYING DATABASE
        #print "SELECT %s FROM edges %s ORDER BY edgeid;" % (','.join(fields),where)
        cur.execute("SELECT %s FROM edges %s ORDER BY edgeid;" % (','.join(fields),where))
        
        #getting fields this can separate out 'as' statements
        fields = [column[0] for column in cur.description]
       
        #this takes the database output puts it in columns then into a dictionary with field names as keys
        out = dict(zip(fields,N.array(zip(*cur.fetchall()))))
        cur.close()
        
        if 'geometry' in out.keys():
            out['lines'] = [ppygis.Geometry.read_ewkb(row) for row in out['geometry']]
            out.pop('geometry')
        
        #print dir(self)
        #print out.keys()
        #self.lines = out['lines']
        
        for key in out.keys():setattr(self,key,out[key])
        
        if len(out.keys())>1:return out
        else:return out[key]
        
        
    def get_attribute(self,attribute):
        if type(getattr(self,attribute))==NoneType:return self.retrieve_fields(attribute)
        else:return getattr(self,attribute)
                            
    def measure_straightenss(self,updatefield=None):

        if type(self.lines)==NoneType:
            #print 'here'
            self.retrieve_fields('geometry')
        if type(self.edgeid)==NoneType:self.retrieve_fields('edgeid')
        
        xs = [N.array([pt.x for pt in ls.points]) for ls in self.lines]
        ys = [N.array([pt.y for pt in ls.points]) for ls in self.lines]

        r2 = []
        for x,y in zip(xs,ys):
            r2.append(stats.linregress(x, y)[2]**2)
        
        if type(updatefield)!=NoneType:
            
            conn = psycopg2.connect(database = 'altimetry', host='localhost')
            cur = conn.cursor()

            for r,eid in zip(r2,self.edgeid):
                cur.execute("UPDATE edges SET %s = %4.3f WHERE edgeid = %i;" % (updatefield,r,eid))
                #print r
            conn.commit()
            cur.close()
        return N.array(r2)

def measure_straightenss(self,updatefield=None):

        if type(self.lines)==NoneType:
            #print 'here'
            self.retrieve_fields('geometry')
        if type(self.edgeid)==NoneType:self.retrieve_fields('edgeid')
        
        xs = [N.array([pt.x for pt in ls.points]) for ls in self.lines]
        ys = [N.array([pt.y for pt in ls.points]) for ls in self.lines]

        r2 = []
        for x,y in zip(xs,ys):
            r2.append(stats.linregress(x, y)[2]**2)
        
        if type(updatefield)!=NoneType:
            
            conn = psycopg2.connect(database = 'altimetry', host='localhost')
            cur = conn.cursor()

            for r,eid in zip(r2,self.edgeid):
                cur.execute("UPDATE edges SET %s = %4.3f WHERE edgeid = %i;" % (updatefield,r,eid))
                #print r
            conn.commit()
            cur.close()
        return N.array(r2)            

imgpath ='/Users/igswahwsmcevan/force/timelapse/AK01_20070818_030132.tagged.jpg'
photoid = None
threshold1=0
threshold2=200
overwriteallphoto = True
overwriteparamrun = False
binarycutoff = 127

thisphoto = 3
print '1'
# a = TimeImage(imgpath)
print '2'
# thres1 = 50
# thres2 = 200
#
# a.Canny1(1100,1300,overwriteparamrun = True,apertureSize=5)
#
#
#for t1 in N.arange(1100,1601,100):
#    for t2 in N.arange(0,401,50):
#        t3 = t1+t2
#        #print t1,t3
#        start_time = time.time()
#        a.Canny1(1100,1300,overwriteparamrun = True,apertureSize=5)
#        print start_time - time.time()
#        a.saveimg(coloredgesby='random',outpath = "/Users/igswahwsmcevan/force/cannyeval_5rad/canny%s_%s.jpg" % (t1,t3))


#f = a.get_lines(asxy=True, negy=False)
#a.saveimg(setlims=[1600,1800,800,800])  

#x,y = a.get_lines(asxy=True)[4].T

#xs,ys = a.get_lines(asxy=True).T

def angular_std(x,y):

    x=N.array(x)
    y=N.array(y)
    rise = (y[1:]-y[:-1])
    rn = (x[1:]-x[:-1])
    angles = N.arctan2(rise,rn)
    return sqrt(abs(moment(angles,2,axis=0)))
    
y= N.random.rand(6)
print angular_std([1,2,3,4,5,6],y)

plt.plot([1,2,3,4,5,6],y)
plt.show()


#plt.show()