import psycopg2
import subprocess as sub
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
import cProfile
import shapefile as shp

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
import pycircstat
from joblib import Parallel, delayed
def querydb(select,asdict=True,aslist=False,returnfields=False):
    
    if aslist==True:asdict=False

    conn = psycopg2.connect(database = 'altimetry', host='localhost')
    cur = conn.cursor()
    
    #QUERYING DATABASE
    cur.execute(select)
    
    #getting fields this can separate out 'as' statements
    fields = [column[0] for column in cur.description]
    
    #this takes the database output puts it in columns then into a dictionary with field names as keys
    if asdict:out = dict(zip(fields,N.array(zip(*cur.fetchall()))))
    else:
        out = N.array(zip(*cur.fetchall()))
        if type(out)==numpy.ndarray:
            if out.size==1:out=out[0]
        
    
    cur.close() 
    
    if returnfields: out = [out, fields] 
    
    return out
    
def alterdb(sql,conn=None,close=True,database = 'altimetry', host='localhost'):

    if conn:
        cur = conn.cursor()
    else:
        conn = psycopg2.connect(database = database, host=host)
        cur = conn.cursor()
        
    #QUERYING DATABASE
    cur.execute(sql)
    if re.search("RETURNING", sql, re.IGNORECASE):
        out = cur.fetchall()
    conn.commit()
    
    if close and not conn:cur.close()
    
    if 'out' in locals():return out

        
def import_raster(filepath,dtmparse = '(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})',cameraid=1,locationid=1):
    
    filename = os.path.basename(filepath)
 
    
    datetxt = N.array(re.findall(dtmparse,filename)[0]).astype(int)
    date = dtm.datetime(datetxt[0],datetxt[1],datetxt[2],datetxt[3],datetxt[4],datetxt[5])
    
    # CHECKING IF FILE HAS BEEN IMPORTED ALREADY 
    if len(querydb("SELECT photoid FROM timelapse WHERE path='%s';" % filepath,aslist=False))!=0:
        print "WARNING: This image has already been imported, import canceled"
        return None
    
    #IMPORTING RASTER FILE
    os.system("/Applications/Postgres.app/Contents/Versions/9.3/bin/raster2pgsql -a -R -f image -Y %s public.timelapse | /Applications/Postgres.app/Contents/Versions/9.3/bin/psql -d altimetry" % filepath)
    
    #FINDING ID OF LAST ENTRY IN TABLE
    #cur.execute("SELECT photoid::int from timelapse order by photoid desc limit 1;")
    photoid = querydb("SELECT photoid::int from timelapse order by photoid desc limit 1;",aslist=True)

    #UPDATING OTHER IMAGE METADATA
    alterdb("UPDATE timelapse SET (cameraid,locationid,date,path) = (%i,%i,'%s','%s') WHERE photoid=%i;" % (cameraid,locationid,re.sub('T',' ', date.isoformat()),filepath,photoid))

    
    #FINDING ID OF THIS IMAGE
    out = querydb("SELECT photoid::int from timelapse order by photoid desc limit 1;")
    
    return out
    
def import_timelapse_dir(folder, select = "*"):
  
    #getting that backslash in there
    if re.search("[^/]$",folder):folder="%s/" % folder
    
    
    filepaths = glob.glob(folder+select)
    
    for f in filepaths:
        out = import_raster(f)
def copydata_fromcsv(*args,**kwargs):
    
    #datatypes = [querydb("SELECT data_type FROM information_schema.columns WHERE table_name='%s' AND column_name='%s';" % (kwargs['table'],c),aslist=True)[0] for c in kwargs['columns']]

    data = zip(*args)
    
    copyformat = "%s\n" % ','.join(kwargs['format'])
    buffer = StringIO.StringIO()
    for line in data:buffer.write(copyformat % line)
    buffer.seek(0)
    
    if 'connection' in kwargs.keys():
        cur = kwargs['connection'].cursor()
    else:
        conn = psycopg2.connect(database = 'altimetry', host='localhost')
        cur = conn.cursor()
        
    cur.copy_from(buffer, kwargs['table'], sep=',', null='\\N', columns=kwargs['columns']) 
    conn.commit() 
    buffer = None
                                                  
class TimeImage:
    def __init__(self, thisphoto):
        if type(thisphoto)==str:
            if os.path.isabs(thisphoto):
                self.imagepath = thisphoto
                self.photoid = querydb("SELECT photoid::int as photoid from timelapse where path = '%s';" % self.imagepath,aslist=True)[0]     #GetSqlData2("SELECT photoid::int as photoid from timelapse where path = '%s';" % self.imagepath)['photoid'][0]
        elif isinstance(thisphoto, (int, long, float)):
            if querydb("SELECT EXISTS(SELECT 1 FROM timelapse WHERE photoid=%i)" % thisphoto,aslist=True):
                self.photoid  = thisphoto
                #print 'here',"SELECT path FROM timelapse WHERE photoid = %s;" % self.photoid,querydb("SELECT path FROM timelapse WHERE photoid = %s;" % self.photoid)['path'][0]
                self.imagepath = querydb("SELECT path FROM timelapse WHERE photoid = %s;" % self.photoid)['path'][0]
            else: raise "This photoid or path cannot be found"
        else:raise "Please provide a valid photoid OR image path"
        self.img = None
        self.edges = None
        
    def get_image(self):
        if type(self.img)==NoneType:
            self.img = cv2.imread(self.imagepath,cv2.IMREAD_COLOR)
            return self.img
        else:
            return self.img

    def Canny1(self, threshold1=0,threshold2=200,overwriteallphoto = True,overwriteparamrun = False,binarycutoff = 127,apertureSize=3):

        #CALCULATING EDGES
        edges = cv2.Canny(self.get_image(),threshold1,threshold2,apertureSize=apertureSize)
        ret,thresh = cv2.threshold(edges,binarycutoff,255,0)  
        contours,hierarchy  = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 


        if len(contours)==0:raise "ERROR: No edges identified under current parameters."
        
        #PREPING THEM TO BE INSERTED AS A GEOMETRY    
        #Prepare data for upload to database
        #parameters used for this run
        params = "Canny1 threshold1:%s threshold2:%s binarycutoff:%s " % (threshold1,threshold2,binarycutoff)
        now = re.sub('T',' ',dtm.datetime.now().isoformat())

        #CHECKING REQUESTS FOR OVERWRITES
        if overwriteallphoto:alterdb("DELETE from edgeparameters WHERE photoid=%s;" % self.photoid)

        if overwriteparamrun:alterdb("DELETE from edgeparameters WHERE photoid=%s AND params='%s';" % (self.photoid,params))

        #INSERTING INTO EDGE PARAMETER TABLE
        lines=[ppygis.LineString([gis.Point(o[0][0],-o[0][1]) for o in c]).write_ewkb() for c in contours]

        paramid = alterdb("INSERT INTO edgeparameters (photoid,params,rundate) VALUES (%i,'%s','%s') RETURNING paramid;" % (self.photoid,params,now))[0][0]

        start_time = time.time()
        
        h1,h2,h3,h4 = zip(*hierarchy[0])
        
        copydata_fromcsv(N.zeros_like(h1)+self.photoid,N.zeros_like(h1)+paramid,h1,h2,h3,h4,lines,columns=['photoid','paramid','presamelvl','nextsamelvl','child','parent','geometry'],table='edges',format = ["%i","%i","%i","%i","%i","%i","%s"])

        
        #for h,l in zip(hierarchy[0],lines):
            #alterdb("INSERT INTO edges (photoid,paramid,presamelvl,nextsamelvl,child,parent,geometry) VALUES (%i,%i,%i,%i,%i,%i,'%s');" % (N.zeros_like(self.photoid)+self.photoid,N.zeros_like(paramid)+paramid,h[0],h[1],h[2],h[3],l.write_ewkb()))
        #print time.time()-start_time        

        #RETURN AN OBJECT that points to this data in the db AND MAKE THIS A METHOD OF A TIMELAPSE OBJECT
        self.edges = Edges(where="photoid = %s" % self.photoid,presamelvl=hierarchy.T[0],nextsamelvl=hierarchy.T[1],child=hierarchy.T[2],parent=hierarchy.T[3],geometry=lines,paramid=paramid)

        return self.edges  
         
    def showimg(self, outpath = None, showedges = True,setlims=None,coloredgesby=None,colorlimits = None, uselines=None,showlines=None):
        import matplotlib.collections as col
        
        fig = plt.figure(figsize=[self.get_image().shape[1]/300.,self.get_image().shape[0]/300.],dpi = 300)
        ax = fig.add_axes([0,0,1,1],frameon=False)
        ax.imshow(self.get_image()[:,:,::-1])
        
        if showedges and type(self.edges)!=NoneType:

            if type(coloredgesby)==NoneType:
                colors='r'
            else:
                
                if type(coloredgesby) == str:
                    if coloredgesby == 'random':
                        colorby = N.random.rand(len(a.get_lines().lines))
                else:
                    mx = coloredgesby.max()
                    colorby = N.where(N.isnan(coloredgesby.astype(float)),mx,coloredgesby)

        
                if colorlimits:
                    norm = mpl.colors.Normalize(vmin=colorlimits[0], vmax=colorlimits[1])
                else:
                    norm = mpl.colors.Normalize(vmin=colorby.min(), vmax=colorby.max())
                print colorby.min(),colorby.max()
                cmap = cm.hot
        
                
                m = cm.ScalarMappable(norm=norm, cmap=cmap)
                colors=m.to_rgba(colorby)  
        
            if not uselines:
                inlines = self.edges.get_lines(format='xy',ys='pos')
                if type(showlines)!=NoneType:
                    w=N.where(showlines)[0]
                    inlines = N.array(inlines)[w]
                    colors = colors[w]
                    
                ln_coll = col.LineCollection(inlines,lw=0.3,color=colors)
            else:
                if type(showlines)!=NoneType:
                    w=N.where(showlines)[0]
                    inlines = N.array(uselines)[w]
                    colors = colors[w]
                else:inlines=uselines
                    
                ln_coll = col.LineCollection(geom_converter(inlines,to_xy=True,negy=True),lw=0.3,color=colors)
            ax.add_collection(ln_coll)
            

            
        if type(setlims)!=NoneType:
            ax.set_xlim(setlims[0],setlims[0]+setlims[2])
            ax.set_ylim(setlims[1],setlims[1]-setlims[3])
            
        plt.draw()
        if type(outpath)==str:
            fig.savefig(outpath,dip=300)
            
        else:
            plt.show()
            
    def add_edges(self,getfields=None):
        if type(self.edges)==NoneType:
            self.edges = Edges(photoid=self.photoid,where="photoid = %s" % self.photoid)
            succeed = self.edges.retrieve_fields(getfields)
            if type(succeed)==NoneType:self.edges=None

            
class Edges:
    def __init__(self,where = None,geometry = None,edgeid=None,photoid =None,presamelvl=None,nextsamelvl=None,child=None,parent=None,selected=None,front=None,paramid=None):

        self.where = where
        self.geometry = geometry
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
        #print getfields
        if type(getfields)==str:fields = re.split('\s*,?\s*',getfields)
        elif type(getfields)==list:fields=getfields
        elif type(getfields)==NoneType:fields = 'geometry'
        else:raise 
        
        if self.where==None:raise "EdgeObject needs a where attribute to run get_fields"
        
        #print 'where',self.where
        #for i,field in enumerate(fields):
        #    #if field == 'edgeid':fields[i]='edgeid::int as edgeid'
        #    #if field == 'geometry':field='lines'
        
        if not re.search('^\s*WHERE',self.where):
            where = "WHERE %s" % self.where
        else:
            where=self.where
        
        #QUERYING DATABASE
        if type(fields)==list:fields = ','.join(fields)
        #print "SELECT %s FROM edges %s ORDER BY edgeid;" % (fields,where)
        out = querydb("SELECT %s FROM edges %s ORDER BY edgeid;" % (fields,where))
        
        if len(out.keys())==0:return
        for key in out.keys():setattr(self,key,out[key])

        if len(out.keys())>1:return out
        else:return out[key]
        
        
    def get_attribute(self,attribute):
        if type(getattr(self,attribute))==NoneType:return self.retrieve_fields(attribute)
        else:return getattr(self,attribute)
        
    def get_lines(self,format=None,ys=None,updateattribute=False):
        #format can be 'xy','ppy',or 'bin'  #ys can be 'neg' or 'pos'  
        
        if type(getattr(self,'geometry'))==NoneType:self.retrieve_fields('geometry')

        if format!=None or ys!=None:

            if format==None:raise "ERROR: if you specify format or ys, you must specify both."
            formatbool = N.array(['xy','ppy','bin'])==format
            if ys=='neg':negbool = True
            elif ys=='pos':negbool=False
            else: raise "ERROR: ys must be either 'neg' or 'pos'"
            
            if updateattribute: self.geometry = geom_converter(self.geometry, to_xy=formatbool[0],to_ppy=formatbool[1], to_bin=formatbool[2],negy=negbool,verbose=False) 
            
            return geom_converter(self.geometry, to_xy=formatbool[0],to_ppy=formatbool[1], to_bin=formatbool[2],negy=negbool,verbose=False) 
            
        else: return self.geometry
        
    def update_db(self,*args,**kwargs):
        
        if type(kwargs['columns'])!=list:cols = [kwargs['columns']]
        else: cols = kwargs['columns'][:]
        
        columns = ','.join(["%s = c.col%i" % (c,i) for c,i in zip(cols,N.arange(len(cols)))])
        columns = re.sub("True",'t',columns)
        columns = re.sub("False",'f',columns)
        
        if type(self.edgeid)==NoneType:self.get_attribute('edgeid')
        
        args2 = list(args)
        args2.append(self.edgeid)
        values = str(zip(*args2))[1:-1]
        values = re.sub("None|nan","NULL",values)

        alterdb("UPDATE edges as t SET \n%s\nFROM (values\n%s\n) AS c(%s)\nWHERE c.col%i=t.edgeid;" % (columns,values,','.join(["col%i" % j for j in N.arange(len(cols)+1)]),i+1))
                                
    def measure_straightness(self,updatefield=None,use_regression=False,use_anglestd=True):

        if use_regression==use_anglestd:raise "ERROR: Please pick either anglestd or regression method"
        if type(self.geometry)==NoneType:
            #print 'here'
            self.retrieve_fields('geometry')
        if type(self.edgeid)==NoneType:self.retrieve_fields('edgeid')
        
        #xys = self.get_lines(format='xy',ys='neg')
        xys = geom_converter(self.retrieve_fields("ST_SimplifyPreserveTopology(geometry,3)"),to_xy=True,negy=True)
        xs=[xy[:,0] for xy in xys]
        ys=[xy[:,1] for xy in xys]
        r2 = []
        
        length = self.retrieve_fields("ST_Length(geometry)")
        
        #print length
        
        for x,y,l in zip(xs,ys,length):

            if use_regression:
                r2.append(stats.linregress(x, y)[2]**2)
            elif use_anglestd:
                r2.append(angular_std(x,y)/l)
        
        if type(updatefield)!=NoneType:
            self.update_db(r2,columns=updatefield)
        
        self.straightness1 = r2
            
            #for r,eid in zip(r2,self.edgeid):
            #    if N.isnan(r):continue
            #    alterdb("UPDATE edges SET %s = %4.3f WHERE edgeid = %i;" % (updatefield,r,eid))

        return N.array(r2)
        
    #def add_front(self, shapefile):
    #        
    #    sf = shp.Reader(shapefile)
    #        
    #    
    #    #HANDLING MULTIPLE POLYGONS...CAN'T DO THAT YET
    #    shapes = sf.shapes()
    #    if len(shapes)>1:raise "ERROR: Shapefile has more than one geometry"
    #    
    #    #print N.array([shapes[0].points])
    #    geom = geom_converter(N.array([shapes[0].points]), to_bin=True,negy=True,verbose=False)
    #    copydata_fromcsv(geom,['t'],[self.photoid], columns=['geometry','front','photoid'],table='edges',format = ["%s","%s","%i"]) 
        
    def get_front(self):
        if type(self.front)==NoneType:
            self.retrieve_attribute("")
                 
def geom_converter(geom, to_ppy=False, to_xy=False,to_bin=False,negy=True,verbose=True):
    if sum([to_ppy,to_xy,to_bin])!=1:raise "ERROR: please select one output format"
    
    #print re.search('|S',geom[2].dtype)
    
    #TESTING TO SEE IF INPUT GEOMETRY IS BINARY
    try:
        if geom[0].dtype.char == 'S':  #  INPUT IS A BINARY GEOMETRY
            print 'binary'
            
            if to_bin:
                if verbose: print 'Input unchanged'
                return geom
            
            ppy=[ppygis.Geometry.read_ewkb(row) for row in geom] 
            
            if to_ppy:
                if verbose: print 'Converting binary to ppy.Linestring'
                return ppy
                
            if to_xy:
                if negy:
                    if ppy[0].points[0].y<0:
                        if verbose: print 'Converting binary to xy. Leaving ys negative.'
                        return [N.array([[pt.x,pt.y] for pt in ls.points]) for ls in ppy]
                    else:
                        if verbose: print 'Converting binary to xy. Reversing ys to negative.'
                        return [N.array([[pt.x,-pt.y] for pt in ls.points]) for ls in ppy] 
                else:
                    if ppy[0].points[0].y<0:
                        if verbose: print 'Converting binary to xy. Reversing ys to positive.'
                        return [N.array([[pt.x,-pt.y] for pt in ls.points]) for ls in ppy]
                    else:
                        if verbose: print 'Converting binary to xy. Leaving ys positive.'
                        return [N.array([[pt.x,pt.y] for pt in ls.points]) for ls in ppy] 
          
    except:pass
    
        #TESTING TO SEE IF INPUT GEOMETRY IS PPYGIS.LINESTRING

    if type(geom[0])==ppygis.LineString:  #  INPUT IS A ppygis GEOMETRY
            print 'ppygis'
            
            if to_ppy:
                if verbose: print 'Input unchanged'
                return geom

            if to_bin:
                #MAKING SURE BINARY VERSION HAS NEGATIVE Y VALUES
                if geom[0].points[0].y>0:
                    for g in geom:
                        for p in g.points:
                            p.y=-p.y  #THIS MIGHT BE A TIME HOG!!!!
                    if verbose: print 'Converting ppy.Linestring to binary. Reversing xys to negative.'
                else:
                    if verbose: print 'Converting ppy.Linestring to binary. Leaving xys negative.'
                return N.array([ppygis.Geometry.write_ewkb(row) for row in geom])

            if to_xy:
                if negy:
                    if geom[0].points[0].y<0:
                        if verbose: print 'Converting ppy.Linestring to xy. Leaving ys negative.'
                        return [N.array([[pt.x,pt.y] for pt in ls.points]) for ls in geom]
                    else:
                        if verbose: print 'Converting ppy.Linestring to xy. Reversing ys to negative.'
                        return [N.array([[pt.x,-pt.y] for pt in ls.points]) for ls in geom] 
                else:
                    if geom[0].points[0].y<0:
                        if verbose: print 'Converting ppy.Linestring to xy. Reversing ys to positive.'
                        return [N.array([[pt.x,-pt.y] for pt in ls.points]) for ls in geom]
                    else:
                        if verbose: print 'Converting ppy.Linestring to xy. Leaving ys positive.'
                        return [N.array([[pt.x,pt.y] for pt in ls.points]) for ls in geom] 
                        
    #TESTING TO SEE IF INPUT GEOMETRY IS XY                     
    if N.array(geom[0][0][0]).dtype==float:  #  INPUT IS A xy GEOMETRY
        
        if geom[0][0,1]<0:
            nowneg = True
        else:
            nowneg = False
            
        if to_xy:

            if negy == nowneg:
                if verbose: print 'Input Unchanged'
                return geom
            if negy != nowneg:

                if verbose and negy: print 'Returning xy. Converting ys to positive.'
                if verbose and not negy: print 'Returning xy. Converting ys to negative.'
                
                out = geom[:]
                for g in out:g[:,1]=-g[:,1]
                return out
                
        if to_ppy:
                
            if negy == nowneg:
                if verbose and negy: print 'Converting xy to ppy.Linestring. Leaving ys negative.'
                if verbose and not negy: print 'Returning ppy.Linestring. Leaving ys positive.'
                ppy=[ppygis.LineString([ppygis.Point(o[0][0],o[0][1]) for o in c]) for c in geom]
            if negy != nowneg: 
                if verbose and negy: print 'Converting xy to ppy.Linestring. Converting ys to negative.'
                if verbose and not negy: print 'Returning ppy.Linestring. Converting ys to positive.'                
                ppy=[ppygis.LineString([ppygis.Point(o[0][0],-o[0][1]) for o in c]) for c in geom]
                
            return ppy
            
           
        if to_bin:
                
            if nowneg:
                if verbose and negy: print 'Converting xy to binary. Leaving ys negative.'
                bi=N.array([ppygis.LineString([ppygis.Point(o[0],o[1]) for o in c]).write_ewkb() for c in geom])
            if not nowneg: 
                if verbose and negy: print 'Converting xy to binary. Converting ys to negative.'
                bi=N.array([ppygis.LineString([ppygis.Point(o[0],-o[1]) for o in c]).write_ewkb() for c in geom])
                
            return bi

def angular_std(x,y):

    if type(x)==list:x=N.array(x)
    if type(y)==list:y=N.array(y)
    
    rise = (y[1:]-y[:-1])
    rn = (x[1:]-x[:-1])
        
    # HERE REMOVING LINES IF THEY ARE THE SAME LINE BUT IN OPPOSITE DIRECTIONS.  NOW LINES GOING BACKWARDS ARE FLIPPED FORWARDS
    rise = N.where(rn<0,-rise,rise)  
    rn = N.abs(rn)
    
    angles = N.arctan2(rise,rn)
    #print angles*180/math.pi
    
    return pycircstat.astd(angles)
    
imgpath ='/Users/igswahwsmcevan/force/timelapse/AK01_20070818_030132.tagged.jpg'
photoid = None
threshold1=0
threshold2=200
overwriteallphoto = True
overwriteparamrun = False
binarycutoff = 127




##sf = shp.Reader("/Users/igswahwsmcevan/Desktop/Untitled.shp")
#    
#
##HANDLING MULTIPLE POLYGONS...CAN'T DO THAT YET
#shapes = sf.shapes()
#nshapes = len(shapes)
#print 'nshapes', nshapes



thisphoto = 3
#import_timelapse_dir("/Users/igswahwsmcevan/force/timelapse/")
a = TimeImage(22)

#a.showimg()
#start_time = time.time()

#a.Canny1(threshold1=1100,threshold2=1300,apertureSize=5)
#print start_time - time.time()
a.add_edges()
#a.edges.add_front("/Users/igswahwsmcevan/Desktop/Untitled.shp")
#a.edges.measure_straightness(updatefield='straightness1',use_regression=True,use_anglestd=False)
#print a.edges.get_lines(format='xy',ys='neg')
#geometry=a.edges.retrieve_fields("ST_SimplifyPreserveTopology(geometry,3)")


a.showimg(coloredgesby=a.edges.retrieve_fields('straightness1'),colorlimits = [0.5,1])
#a.edges.get_attribute('straightness1')
#a.edges.get_attribute('edgeid')
#a.edges.update_db(a.edges.straightness1,columns=['straightness1'])
#b = copydata_fromcsv([1,2,3,4,5],[6,7,8,9,0],[6,7,8,9,0],columns=['geometry','front','selected'],table='edges',format = ["'%s'","'%s'","%s"])

#e,length = a.get_lines(modify_geometry="ST_Simplify(geometry,1) as geometry",getfields="ST_Length(geometry)::real as length")


#eid = querydb("SELECT edgeid from edges where photoid=18 limit 1;",asdict=True,aslist=True)

#a.Canny1(1100,1300,overwriteparamrun = True,apertureSize=5)
#print 'here'
#straight = e.measure_straightness(use_regression=False,use_anglestd=True)
#
#
##
##for t1 in N.arange(1100,1601,100):
##    for t2 in N.arange(0,401,50):
##        t3 = t1+t2
##        #print t1,t3
##        start_time = time.time()
##        a.Canny1(1100,1300,overwriteparamrun = True,apertureSize=5)
##        print start_time - time.time()
##        a.saveimg(coloredgesby='random',outpath = "/Users/igswahwsmcevan/force/cannyeval_5rad/canny%s_%s.jpg" % (t1,t3))
#
#
#f = querydb("SELECT geometry from edges limit 3;")['geometry']
#print re.search('|S',f[2].dtype)=geom_converter(f, to_ppy=True, to_xy=False,to_bin=False,negy=False)


#colors = straight#a.edges.get_attribute('straightness1').astype(float)
#mx = N.compress(~N.isnan(colors),colors).max()
#colors = N.where(N.isnan(colors),mx,colors)
#
#e.lines
#a.saveimg(coloredgesby=colors,colorlimits=[1.,1.5],uselines=e.geometry)#setlims=[1600,1800,800,800])  

#x,y = a.get_lines(asxy=True)[4].T

#xs,ys = a.get_lines(asxy=True).T


    
