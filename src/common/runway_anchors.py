"""Fixed M10 ORD runway coordinates and read-only ENU alignment geometry."""
import numpy as np

# SYGARD_MOTION_PLAN.md M10 fixed OurAirports/public-domain coordinates.
RUNWAYS=[
 ('4L/22R','04L',41.9817009,-87.9139023,'22R',41.9975014,-87.8964005),
 ('4R/22L','04R',41.9533005,-87.8993988,'22L',41.9698982,-87.8797989),
 ('9C/27C','09C',41.98831,-87.931584,'27C',41.988326,-87.890211),
 ('9L/27R','09L',42.0028305,-87.9266739,'27R',42.0028305,-87.8990860),
 ('9R/27L','09R',41.9838982,-87.9183502,'27L',41.9838982,-87.8890533),
 ('10C/28C','10C',41.9656982,-87.9315033,'28C',41.9656982,-87.8917999),
 ('10L/28R','10L',41.9690018,-87.9315033,'28R',41.9691010,-87.8836975),
 ('10R/28L','10R',41.9571991,-87.9279022,'28L',41.9572449,-87.9002914),
]
EXPECTED={'10L/28R':1474,'9C/27C':1052,'9L/27R':849,'10C/28C':744,
          '4R/22L':403,'9R/27L':219,'10R/28L':171,'4L/22R':0}


def geodetic_to_enu(latitude,longitude,altitude,origin):
    """WGS84 ellipsoid, east/north/up metres; matches reference sample to1e-9m."""
    def ecef(lat,lon,alt):
        lat,lon=np.radians([lat,lon]);e2=.0066943799901413165
        n=6378137/np.sqrt(1-e2*np.sin(lat)**2)
        return np.array([(n+alt)*np.cos(lat)*np.cos(lon),(n+alt)*np.cos(lat)*np.sin(lon),
                         (n*(1-e2)+alt)*np.sin(lat)])
    lat,lon=np.radians([origin['latitude_deg'],origin['longitude_deg']])
    rotation=np.array([[-np.sin(lon),np.cos(lon),0],
        [-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)],
        [np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]])
    return rotation@(ecef(latitude,longitude,altitude)-ecef(origin['latitude_deg'],origin['longitude_deg'],origin['altitude_m']))


def fixed_endpoints(origin):
    """XY at origin altitude per advisor clarification; Z fitted separately on train."""
    rows=[]
    for name,a,lat_a,lon_a,b,lat_b,lon_b in RUNWAYS:
        pa=geodetic_to_enu(lat_a,lon_a,origin['altitude_m'],origin)
        pb=geodetic_to_enu(lat_b,lon_b,origin['altitude_m'],origin)
        for ident,lat,lon,point,other in [(a,lat_a,lon_a,pa,pb),(b,lat_b,lon_b,pb,pa)]:
            delta=other-point
            rows.append(dict(anchor_id=len(rows),runway=name,ident=ident,latitude_deg=lat,longitude_deg=lon,
                x_m=float(point[0]),y_m=float(point[1]),z_m=None,
                inbound_heading_rad=float(np.arctan2(delta[1],delta[0]))))
    return rows


def runway_projection(position,a,b):
    a,b=np.asarray(a),np.asarray(b);delta=b-a;length=np.linalg.norm(delta)
    unit=delta/length;offset=np.asarray(position)[...,:2]-a
    along=(offset*unit).sum(axis=-1)
    perpendicular=np.abs(offset[...,0]*unit[1]-offset[...,1]*unit[0])
    return along,perpendicular,float(length),float(np.arctan2(unit[1],unit[0]))


def alignment_masks(position,heading,valid,on_ground,a,b):
    """Exact corrected M10 masks, ported from the read-only reference script."""
    along,perp,length,axis=runway_projection(position,a,b)
    ground=valid & on_ground & (perp<40) & (along>0) & (along<length)
    ground &= np.abs(np.cos(heading-axis))>.97
    air=valid & ~on_ground & (position[...,2]<500) & (perp<120)
    air &= np.abs(np.cos(heading-axis))>.97
    air &= ((along>-6000)&(along<0))|((along>length)&(along<length+6000))
    nearby_ground=valid & on_ground & (perp<=100) & (along>=0) & (along<=length)
    return ground,air,nearby_ground
