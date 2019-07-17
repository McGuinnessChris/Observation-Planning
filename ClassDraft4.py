#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pytz
from tzlocal import get_localzone

from tabulate import tabulate

import astroplan
from astroplan import FixedTarget
from astroplan import Observer
from astroplan import constraints
from astroplan import (AltitudeConstraint,
                       AirmassConstraint,
                       AtNightConstraint,
                       MoonSeparationConstraint,
                       MoonIlluminationConstraint)
from astroplan import (is_observable,
                       is_always_observable,
                       months_observable)
from astroplan import observability_table
from astroplan.scheduling import Transitioner
from astroplan.scheduling import SequentialScheduler
from astroplan.scheduling import Schedule
from astroplan import ObservingBlock
from astroplan.constraints import TimeConstraint
from astroplan.plots import plot_schedule_airmass

import astropy
from astropy.time import Time
from astropy.coordinates import EarthLocation
from astropy.coordinates import SkyCoord
from astropy import coordinates
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table, Column

from timezonefinder import TimezoneFinder
tf = TimezoneFinder()

# from astroplan import download_IERS_A
# download_IERS_A()


################################################################################
################################################################################
################################################################################

########## Some necessary functions ############################################

# made names long and included "Default" so there is no chance of me
# accidentally making another variable with the same name in a
# class later
UTC = pytz.timezone('UTC')

################################################################################
########## Dublin Info #########################################################

Dublin = Observer(location = EarthLocation.from_geodetic(
                  -6*u.deg,
                  53.35*u.deg,
                  6*u.m),
                  name = "Dublin",
                  timezone = "Europe/Dublin")


twilight_evening_astronomical_UTC_Default = (
                        Dublin.twilight_evening_nautical(
                        Time(dt.datetime.now().astimezone(UTC)),
                        which='next') )
eveningDublin = twilight_evening_astronomical_UTC_Default.strftime('%Y-%m-%d %H:%M')

# twilight_evening_astronomical_Dublin_Default = (
#                 Dublin.astropy_time_to_datetime(
#                 twilight_evening_astronomical_UTC_Default).strftime('%Y-%m-%d %H:%M') )
# eveningDublin = twilight_evening_astronomical_Dublin_Default


twilight_morning_astronomical_UTC_Default = (
                        Dublin.twilight_morning_nautical(
                        Time(twilight_evening_astronomical_UTC_Default),
                        which='next') )
morningDublin = twilight_morning_astronomical_UTC_Default.strftime('%Y-%m-%d %H:%M')

# twilight_morning_astronomical_Dublin_Default = (
#                 Dublin.astropy_time_to_datetime(
#                 twilight_morning_astronomical_UTC_Default).strftime('%Y-%m-%d %H:%M') )
# morningDublin = twilight_morning_astronomical_Dublin_Default

################################################################################
########## Veritas Info ########################################################

VeritasLocationDefault = Observer(location = EarthLocation.from_geodetic(
                                  -110.95019*u.deg,
                                  31.675*u.deg,
                                  1268*u.m),
                                  name = "Veritas",
                                  timezone = "America/Phoenix")

twilight_evening_astronomical_UTC_Default = (
                        VeritasLocationDefault.twilight_evening_astronomical(
                        Time(dt.datetime.now().astimezone(UTC)),
                        which='next') )
startDefault = twilight_evening_astronomical_UTC_Default.strftime('%Y-%m-%d %H:%M')

# twilight_evening_astronomical_at_observatory_Default = (
#                 VeritasLocationDefault.astropy_time_to_datetime(
#                 twilight_evening_astronomical_UTC_Default).strftime('%Y-%m-%d %H:%M') )
# startDefault = twilight_evening_astronomical_at_observatory_Default


twilight_morning_astronomical_UTC_Default = (
                        VeritasLocationDefault.twilight_morning_astronomical(
                        Time(twilight_evening_astronomical_UTC_Default),
                        which='next') )
endDefault = twilight_morning_astronomical_UTC_Default.strftime('%Y-%m-%d %H:%M')

# twilight_morning_astronomical_at_observatory_Default = (
#                 VeritasLocationDefault.astropy_time_to_datetime(
#                 twilight_morning_astronomical_UTC_Default).strftime('%Y-%m-%d %H:%M') )
# endDefault = twilight_morning_astronomical_at_observatory_Default

################################################################################
__all__ = ['Observer', 'Times', 'Targets']


################################################################################
################################################################################


### Most of Astroplan works using Observer objects,
### which are made from EarthLocation objects. EarthLocation
### can be configured to work using either the
### names of locations or their coordinates if they are not listed.
### Using this information, the following function will always return an
### Observer object assigned to the variable "observer".
### The function will default to Veritas if no location is given.


### TimezoneFinder and pytz both share a common list of timezone names.
### This common list is also used by Astropy and Astroplan, however
### timezones are not automatically assigned by Astropy or Astroplan.
### Because of this, this part of the script also creates a variable called
### observatory_timezone which will be used in many calculations later.





class Location:
    """
    The location of the observer or the observatory being used will always
    default to the coordinates for Veritas. To use another location the
    coordinates must be given as longitude, latitude and elevation.
    Alternatively the name ofanother can be accepted if it is part of the
    list of accepted observatories.This list can be returned if required.
    """

#################################################################

    """
    Parameters
    ----------
    name : string
        Name of observatory. Can check list of names using observatory_list().

    longitude : interger or float
        Observer or observatory longitude

    latitude : integer or float
        Observer or observatory latitude

    elevation : integer or float
        Configuration metadata

    """

    def __init__(self , name , longitude , latitude , elevation):
        self.name = name
        self.longitude = longitude
        self.latitude = latitude
        self.elevation = elevation


    def observatory_list():
        return astropy.coordinates.EarthLocation.get_site_names()


    def location_from_name(name):   # uses site name to find coordinates
                                    # and time zone of named observatory
        observer = EarthLocation.of_site(name)
        observatory_timezone = tf.closest_timezone_at(
                                        lng=float(observer.geodetic[0]/u.deg),
                                        lat=float(observer.geodetic[1]/u.deg))

        return observer , observatory_timezone



    def location_from_coordinates(longitude,
                                  latitude,
                                  elevation):
        observer = Observer(location = EarthLocation.from_geodetic(
        longitude*u.deg,
        latitude*u.deg,
        elevation*u.m),
        name = "Observer Location",
        timezone = tf.closest_timezone_at(lng = longitude,
                                  lat = latitude))
        observatory_timezone = tf.closest_timezone_at(lng = longitude,
                                                      lat = latitude)
        observatory_timezone = pytz.timezone(observatory_timezone)

        return observer , observatory_timezone



    def Veritas_location():
        observer = Observer(location = EarthLocation.from_geodetic(
        -110.95019*u.deg,
        31.675*u.deg,
        1268*u.m),
        name = "Veritas",
        timezone = "America/Phoenix")

        observatory_timezone = tf.closest_timezone_at(lng = -110.95019,
                                                      lat = 31.675)
        observatory_timezone = pytz.timezone(observatory_timezone)

        return observer , observatory_timezone


################################################################################
################################################################################


class Time_window:
    """
    Creates datetime.datetime objects.
    This class is used to specify the time window for any observation.
    The time window is constructed from given start and end times and the
    class can return the current time in UTC, local and observatory timezones,
    as well as the observation start and end times in UTC, local and
    observatory timezones. Can also be used to split the time window into
    a number of sections of equal time length for other calculations.
    Can also return local time and timezone if not known by the user.
    """

#################################################################

    """
    Parameters
    ----------
    start : string
        Takes the observation start time (UTC Timezone) as an argument in the
        format '%YYYY-%MM-%DD %HH:%MM'.

    end : string
        Takes the observation end time (UTC Timezone) as an argument in the
        format '%YYYY-%MM-%DD %HH:%MM'.

    sections : interger
        This interger number can be used to divide the time window into
        sections of equal time length.

    timezone : pytz timezone
        Takes a pytz timezone like those returned by the Location class.
    """

    def __init__(self , start , end):
        self.start = start
        self.end = end

    def local_time():
        current_time = dt.datetime.now()
        local_timezone = get_localzone()
        return current_time.astimezone(local_timezone)

    def start(start):
        UTC = pytz.timezone('UTC')
        return dt.datetime.strptime(start , '%Y-%m-%d %H:%M').astimezone(UTC)

    def end(end):
        UTC = pytz.timezone('UTC')
        return dt.datetime.strptime(end , '%Y-%m-%d %H:%M').astimezone(UTC)

    """
    local_to_UTC information:
    this function will calculate the local time of the user and return their
    current time converted to the UTC timezone
    """

    def local_to_UTC():
        local_timezone = get_localzone()
        current_time = dt.datetime.now().astimezone(local_timezone)
        UTC = pytz.timezone('UTC')
        return current_time.astimezone(UTC)

    """
    observatory_to_UTC information:
    this function will take a pytz timezone as returned by the Location class
    to convert a given time in the observatory local timezone to time
    in the UTC timezone
    """

    def observatory_to_UTC(start , end , timezone):
        timezone = pytz.timezone(timezone)
        UTC = pytz.timezone('UTC')
        start = dt.datetime.strptime(start,
                                     '%Y-%m-%d %H:%M').astimezone(timezone)
        end = dt.datetime.strptime(end,
                                     '%Y-%m-%d %H:%M').astimezone(timezone)
        return start.astimezone(UTC) , end.astimezone(UTC)

    """
    UTC_to_observatory information:
    this function will take a pytz timezone as returned by the Location class
    to convert a given time in UTC time to the local time at the observatory/
    observer location
    """

    def UTC_to_observatory(start , end , timezone):
        timezone = pytz.timezone(timezone)
        UTC = pytz.timezone('UTC')
        start = dt.datetime.strptime(start , '%Y-%m-%d %H:%M').astimezone(UTC)
        end = dt.datetime.strptime(end , '%Y-%m-%d %H:%M').astimezone(UTC)
        return start.astimezone(timezone) , end.astimezone(timezone)

    """
    time_range information:
    time_range deals with functions like is_ever_observable, which require
    a start and end time specified in a Time() format
    """

    def time_range(start = startDefault , end = endDefault):
        UTC = pytz.timezone('UTC')
        return Time([dt.datetime.strptime(start , '%Y-%m-%d %H:%M'),
                           dt.datetime.strptime(end , '%Y-%m-%d %H:%M')])

    """
    time_window information:
    time_window defines a start and end time for the observation and divides
    it into a number of equal sections, defaulting to 3 sections.
    The 3 divisions return a list of three time values, the start time, the end
    end time and the time halfway between them.
    """

    def time_window(start = startDefault , end = endDefault , section = 3):
        return ( dt.datetime.strptime(start , '%Y-%m-%d %H:%M')
         + (dt.datetime.strptime(end , '%Y-%m-%d %H:%M')
          - dt.datetime.strptime(start , '%Y-%m-%d %H:%M'))
          * np.linspace(0 , 1 , section) )


################################################################################
################################################################################


class sun_moon_info():
    """
    This class returns information about the sun and moon, including their
    rise and set times, and moon phase and illumination at the specified
    observation start time. Requires an Observer object, which are returned
    by the Location class.
    """

#################################################################

    """
    Parameters
    ----------
    time : string (format '%YYYY-%MM-%DD %HH:%MM')
        Functions in this class must have times given in UTC time, not the
        local observatory/observer time.
        Takes datetime.datetime objects (these are returned by the Time_window
        class).

    start : string (format '%YYYY-%MM-%DD %HH:%MM')
        Functions in this class must have times given in UTC time, not the
        local observatory/observer time.
        Takes datetime.datetime objects (these are returned by the Time_window
        class).

    end : string (format '%YYYY-%MM-%DD %HH:%MM')
        Functions in this class must have times given in UTC time, not the
        local observatory/observer time.
        Takes datetime.datetime objects (these are returned by the Time_window
        class).

    observer : Astroplan Observer object
        Takes an AstroPlan observer object. These are returned by the Location
        class.

    timezone : pytz timezone
        Takes pytz timezones. These are returned by the Location class.
    """

    def __init__(self , start , end , observer , timezone):
        self.start = start
        self.end = end
        self.observer = observer
        self.timezone = timezone

    """
    sun_info_local information:
    returns the next sunrise and sunset times after a given time for the
    user location
    return order is set time then rise time
    """

    def sun_info_UTC(observer , time):
        local_timezone = get_localzone()
        time = ( dt.datetime.strptime(time,
                '%Y-%m-%d %H:%M').astimezone(local_timezone) )
        return ( observer.sun_set_time(Time(time) , which = 'next').iso ,
                 observer.sun_rise_time(Time(time) , which = 'next').iso )

    """
    sun_info_observatory information:
    returns the next sunrise and sunset times after a given time for the
    observatory location in UTC Time
    return order is set time then rise time
    """

    def sun_info_observatory(observer , time):
        UTC = pytz.timezone('UTC')
        time = ( dt.datetime.strptime(time,
                '%Y-%m-%d %H:%M').astimezone(UTC) )
        set = observer.sun_set_time(Time(time) , which = 'next')
        rise = observer.sun_rise_time(Time(time) , which = 'next')

        set = observer.astropy_time_to_datetime(set).strftime('%Y-%m-%d %H:%M')
        rise = observer.astropy_time_to_datetime(rise).strftime('%Y-%m-%d %H:%M')
        ### .iso changes the format returned from a time object
        ### in a strange unit to a legible string with date and time
        return set , rise#set.iso , rise.iso

    """
    moon_info_local information:
    returns the next moonrise and moonset times after a given time for the
    user location, along with moon altitude, azimuth, phase and illumination
    at the given time

    return order is set time, rise time, altitude, azimuth, phase, illumination

    AstroPlan calculates moon phase in radians for some reason
    the conversion here is between 1 and 0 with 1 being full moon and
    0 being a new moon
    """

    def moon_info_UTC(observer , time):
        fmt = '%Y-%m-%d %H:%M'
        local_timezone = get_localzone()
        time = Time( dt.datetime.strptime(time,
                '%Y-%m-%d %H:%M').astimezone(local_timezone) )
        altaz = observer.moon_altaz(time)
        phase = observer.moon_phase(Time(time))
        illumination = observer.moon_illumination(Time(time))
        set = observer.moon_set_time(Time(time) , which = 'next')
        rise = observer.moon_rise_time(Time(time) , which = 'next')
        ### .iso changes the format returned from a time object
        ### in a strange unit to a legible string with date and time
        return ( set.iso,
                 rise.iso,
                 "%.3f" %((altaz).alt /u.deg) *u.deg,
                 "%.3f" %((altaz).az /u.deg) *u.deg,
                 "%.3f" %(1 - (phase / u.rad) / np.pi),
                 "%.3f" %illumination )

    """
    moon_info_observatory information:
    returns the next moonrise and moonset times after a given time for the
    observatory/observer location, along with moon altitude, azimuth, phase
    and illumination at the given time

    return order is set time, rise time, altitude, azimuth, phase, illumination

    AstroPlan calculates moon phase in radians for some reason
    the conversion here is between 1 and 0 with 1 being full moon and
    0 being a new moon
    """

    def moon_info_observatory(observer , time):
        UTC = pytz.timezone('UTC')
        time = Time( dt.datetime.strptime(time,
                '%Y-%m-%d %H:%M').astimezone(UTC) )
        altaz = observer.moon_altaz(time)
        phase = observer.moon_phase(Time(time))
        illumination = observer.moon_illumination(Time(time))
        set = observer.moon_set_time(Time(time) , which = 'next')
        rise = observer.moon_rise_time(Time(time) , which = 'next')
        ### .iso changes the format returned from a time object
        ### in a strange unit to a legible string with date and time
        return ( set.iso,
                 rise.iso,
                 "%.3f" %((altaz).alt).value *u.deg,
                 "%.3f" %((altaz).az /u.deg) *u.deg,
                 "%.3f" %(1 - (phase / u.rad) / np.pi),
                 "%.3f" %illumination )

    """
    twilight_evening_UTC information:
    Returns the time for evening astronomical twilight at the observatory
    in UTC time. Astroplan takes astronomical twilight to be at sun
    elevation of -18 degrees.
    .iso is used to change the format of the Time object from something
    strange into a legible string with the date and time.
    """

    def twilight_evening_UTC(observer , time):
        UTC = pytz.timezone('UTC')
        time = Time( dt.datetime.strptime(time,
                '%Y-%m-%d %H:%M').astimezone(UTC) )
        local_twilight_astronomical = observer.twilight_evening_astronomical(
                                        time , which='next')
        return local_twilight_astronomical.iso


    """
    twilight_morning_UTC information:
    Returns the time for morning astronomical twilight at the observatory
    in UTC time. Astroplan takes astronomical twilight to be at sun
    elevation of -18 degrees.
    .iso is used to change the format of the Time object from something
    strange into a legible string with the date and time.
    """

    def twilight_morning_UTC(observer , time):
        UTC = pytz.timezone('UTC')
        time = Time( dt.datetime.strptime(time,
                '%Y-%m-%d %H:%M').astimezone(UTC) )
        local_twilight_astronomical = observer.twilight_morning_astronomical(
                                        time , which='next')
        return local_twilight_astronomical.iso


    """
    twilight_evening_observatory information:
    The timezone argument should be the observatory timezone in pytz format
    as is returned by the Location class.
    Returns the time for evening astronomical twilight at the observatory
    in observatory time. Astroplan takes astronomical twilight to be at
    sun elevation of -18 degrees.
    .iso is used to change the format of the Time object from something
    strange into a legible string with the date and time.
    """

    def twilight_evening_observatory(observer , time, timezone):
        timezone = pytz.timezone(timezone)
        UTC = pytz.timezone('UTC')
        time = Time( dt.datetime.strptime(time,
                '%Y-%m-%d %H:%M').astimezone(UTC) )
        observatory_twilight = (
        observer.twilight_evening_astronomical(time , which='next') )
        observatory_twilight = dt.datetime.strptime(observatory_twilight.iso,
                                        '%Y-%m-%d %H:%M:%S.%f')
        observatory_twilight = (
        observatory_twilight.astimezone(timezone).strftime('%Y-%m-%d %H:%M') )

        return observatory_twilight


    """
    twilight_morning_observatory information:
    The timezone argument should be the observatory timezone in pytz format
    as is returned by the Location class.
    Returns the time for morning astronomical twilight at the observatory
    in observatory time. Astroplan takes astronomical twilight to be at
    sun elevation of -18 degrees.
    .iso is used to change the format of the Time object from something
    strange into a legible string with the date and time.
    """

    def twilight_morning_observatory(observer , time, timezone):
        timezone = pytz.timezone(timezone)
        UTC = pytz.timezone('UTC')
        time = Time( dt.datetime.strptime(time,
                '%Y-%m-%d %H:%M').astimezone(UTC) )
        observatory_twilight = (
        observer.twilight_morning_astronomical(time , which='next') )
        observatory_twilight = dt.datetime.strptime(observatory_twilight.iso,
                                        '%Y-%m-%d %H:%M:%S.%f')
        observatory_twilight = (
        observatory_twilight.astimezone(timezone).strftime('%Y-%m-%d %H:%M') )

        return observatory_twilight


################################################################################
################################################################################


class targets():
    """
    This class is used to initialize target objects such as stars and galaxies.
    Astroplan creates FixedTarget objects of the given targets which can then
    be operated on by later functions. Some of those functions are also used in
    this class to calculate targe rise and set times.
    Initializing can be done using the target's name or manually inputting the
    right ascension and declination of the target object. Lists of targets will
    also be acceptable.
    """

    """
    Astropy and Astroplan use FixedTarget objects to to calculations
    using the locations of desired targets. The coordinates of these
    FixedTarget objects are first defined as SkyCoord objects.
    Some functions within Astroplan and Astropy take FixedTarget objects
    and some take SkyCoord objects. To further complicate things, some
    functions in Astroplan can take int or float type arguments while others
    only take strings or require correct Astropy Units to be specified for
    the arguments.
    """

#################################################################

    """
    Parameters
    ----------
    name : string
        The name of the desired target, given as a string. If the target is in
        one of the surveys supported by Astroplan, a FixedTarget object can
        be make from just the target's name.

    ra : interger or float
        Target Right Ascension given as a float (in degrees not hours,
        minutes and seconds).

    dec : interger or float
        Target Declination given as a float (in degrees not deg, arcmin and
        arcsec).

    file : .txt file
        A .txt file with either the names of each target or the name and
        coordinates of each target.

    targets : FixedTarget object
        FixedTarget objects are created by the first four functions in this
        class. These can then be used in the later functions to perform
        calculations such as target rise and set times.

    observer : Observer object
        Observer objects are created using the Location class. They initialize
        the location at which the observation is being performed. In this class
        they will be used to calculate target rise and set times at the
        observation location.

    time : string
        Takes a time in the format '%YYYY-%MM-%DD %HH:%MM'. Calculations such
        as target rise and set times will be calculated for the next rise or
        set time after the given time.

    time_window : a time_window object created using Time_window.time_window
        Takes times in the format '%YYYY-%MM-%DD %HH:%MM'. Target-moon angular
        separation will be calculated for a range of times within this
        time_window.
    """

    def __init__(self , name , ra , dec, file , observer , targets , time ,
                 time_window):
        self.name = name
        self.ra = ra
        self.dec = dec
        self.file = file
        self.targets = targets
        self.observer = observer
        self.time = time
        self.time_window = time_window


    """
    target_from_name information:
    Takes string for the name, so must be in format 'name'.

    Example:
    orion = targets.target_from_name('orion')
    """

    def target_from_name(name):
        return FixedTarget.from_name(name)


    def target_from_coordinates(name , ra , dec):
        coord = SkyCoord(ra = ra*u.deg , dec = dec*u.deg)
        return FixedTarget(coord = coord , name = name)

    """
    target_list_names information:
    Takes a single column of target names, titled name. Names must be in the
    format "name".

    Example:
    targets = targets.target_list_names('/home/heap/Documents/Test1.txt')

        Inside Test1.txt:
            name
            "polaris"
            "Vega"
            "crab nebula"
    """

    def target_list_names(file):
        targets = Table.read(file , format = 'ascii')
        return [FixedTarget.from_name(list(targets[i])[0])
                                     for i in range(np.size(targets))]


    """
    target_list_names information:
    Takes a three columns of data; target name, right ascension and
    declination. Names must be in the format "name" and ra and dec can be as
    intergers or float.

    Example:
    targets = targets.target_list_names('/home/heap/Documents/Test1.txt')

        Inside Test2.txt:
            name ra_degrees dec_degrees
            "Polaris" 37.95456067 89.26410897
            "Vega" 279.234734787 38.783688956
            "Crab Nebula" 83.633083 22.0145
    """

    def target_list_coordinates(file):
        targets = ascii.read(file)
        return [FixedTarget(coord = SkyCoord(ra = ra*u.deg,
                                             dec = dec*u.deg),
                                             name = name)
                                            for name , ra , dec in targets]


    """
    target_rise_times information:
    Takes initialized targets (targets made into FixedTarget objects).
    Calculates the next rise time for the target.

    If a target does not rise within 24 hours of the specified time then
    Astroplan returns an warning, labelled TargetAlwaysUpWarning or
    TargetNeverUpWarning.This then triggers a warning "dubious year (Note 5)
    which sets the target rise time to '-4715-02-28 12:00:00.000'
    for some reason.
    """

    def target_rise_times(observer , targets , time):
        rise_time = []
        if np.size(targets) == 1:
            rise = observer.target_rise_time(time,
                                             targets,
                                             which="next")
            if rise.iso == '-4715-02-28 12:00:00.000':
                    rise_time.append("Doesn't rise")
            else:
                rise_time.append(rise.iso)

        else:
            for i in range(np.size(targets)):

                rise = observer.target_rise_time(time,
                                                 targets[i],
                                                 which="next")
                if rise.iso == '-4715-02-28 12:00:00.000':
                        rise_time.append("Doesn't rise")
                else:
                    rise_time.append(rise.iso)
        return rise_time


    """
    target_set_times information:
    Takes initialized targets (targets made into FixedTarget objects).
    Calculates the next rise time for the target.

    If a target does not set within 24 hours of the specified time then
    Astroplan returns an warning, labelled TargetAlwaysUpWarning
    TargetNeverUpWarning.This then triggers a warning "dubious year (Note 5)
    which sets the target set time to '-4715-02-28 12:00:00.000'
    for some reason.
    """

    def target_set_times(observer , targets , time):
        set_time = []
        if np.size(targets) == 1:
            set = observer.target_set_time(time,
                                             targets,
                                             which="next")
            if set.iso == '-4715-02-28 12:00:00.000':
                    set_time.append("Doesn't set")
            else:
                set_time.append(set.iso)

        else:
            for i in range(np.size(targets)):

                set = observer.target_set_time(time,
                                                 targets[i],
                                                 which="next")
                if set.iso == '-4715-02-28 12:00:00.000':
                        # rise_time.append("Doesn't cross horizon during this time")
                    set_time.append("Doesn't set")
                else:
                    set_time.append(set.iso)
        return set_time






    def target_moon_separation(observer , targets , time_window):
        moon_coords = coordinates.get_moon(Time(time_window),
                                           observer.location)
        target_coords = []

        if np.size(targets) == 1:
            object = SkyCoord(targets[0].ra , targets[0].dec)
            target_coords.append(object)

            separations = []
            moons = []

            for i in range(np.size(moon_coords)):
                    sep = target_coords[0].separation(
                          SkyCoord(moon_coords[i].ra , moon_coords[i].dec))
                    moons.append(sep)
            separations.append(moons)
            separations = separations[0]
            #returns moons to being empty
            moons = []

        else:
            for i in range(np.size(targets)):
                object = SkyCoord(targets[i].ra , targets[i].dec)
                target_coords.append(object)

            separations = []
            moons = []

            for i in range(np.size(target_coords)):
                targ = target_coords[i]
                for j in range(np.size(moon_coords)):
                    sep = targ.separation( SkyCoord(moon_coords[j].ra,
                                                    moon_coords[j].dec) )
                    moons.append(sep)
                separations.append(moons)
                #returns moons to being empty
                moons = []

        return separations







################################################################################
################################################################################



################################################################################
########## Defining a custom moon phase constraint #############################
########## based on the Astroplan constraint format ############################
################################################################################


class MoonPhaseConstraint(astroplan.constraints.Constraint):
    """
    Constrain acceptable moon phase to within a maximum and minimum fraction.
    """

#################################################################

    def __init__(self , observer , times , minphase , maxphase , ephemeris=None):

        """
        Parameters
        ----------
        observer : Astroplan Observer object
            Takes an AstroPlan observer object. These are returned by the Location
            class.

        times : string
            Takes a time in the format '%YYYY-%MM-%DD %HH:%MM'. Calculations such
            as target rise and set times will be calculated for the next rise or
            set time after the given time.

        min : `~astropy.units.Quantity` or `None` (optional)
            Minimum acceptable moon phase for observation.
            `None` indicates no limit.

        max : `~astropy.units.Quantity` or `None` (optional)
            Maximum acceptable moon phase for observation.
            `None` indicates no limit.

        ephemeris : str, optional
            Ephemeris to use.  If not given, use the one set with
            ``astropy.coordinates.solar_system_ephemeris.set`` (which is
            set to 'builtin' by default).
        """

        self.observer = observer
        self.times = times
        self.minphase = minphase
        self.maxphase = maxphase
        self.ephemeris = ephemeris

    def compute_constraint(self, times, observer, targets):
        moonphase = self.observer.moon_phase(self.times)

        ### converts moon phase from full moon = 0 radians,
        ### new moon = pi radians to full moon = 1,
        ### new moon = 0. New anything inbetween is calculated as a float.
        moonphase = (1 - (moonphase / u.rad) / np.pi)

        if self.minphase is None and self.maxphase is not None:
            mask = self.maxphase >= moonphase
        elif self.maxphase is None and self.minphase is not None:
            mask = self.minphase <= moonphase
        elif self.minphase is not None and self.maxphase is not None:
            mask = ((self.minphase <= moonphase) &
                    (moonphase <= self.maxphase))
        else:
            raise ValueError("No max and/or min specified in "
                             "MoonPhaseConstraint.")
        return mask



################################################################################
################################################################################
################################################################################

class limits():
    """
    This class will be used to define observation constraints. There is already
    a class called Constraints in Astroplan, which is why this one is called
    limits. All values will default to Veritas standard constraints if not
    specified.
    """

### will include AltitudeConstraint, AirmassConstraint, AtNightConstraint,
### MoonSeparationConstraint, MoonIlluminationConstraint and maybe
### a custom Moon Phase Constraint

#################################################################

    """
    Parameters
    ----------
    observer : Astroplan Observer object
        Takes an AstroPlan observer object. These are returned by the Location
        class.

    times : string
        Takes a time in the format '%YYYY-%MM-%DD %HH:%MM'. Calculations such
        as target rise and set times will be calculated for the next rise or
        set time after the given time.

    altmin : interger or float (units = degrees)
        Defines the minimum angle of elevation acceptable for observation from
        the observatory of any target.

    altmax : interger or float (units = degrees)
        Defines the maximum angle of elevation acceptable for observation from
        the observatory of any target.

    airmass : interger or float
        An airmass of n requires the airmass be “better than n”.

    maxmoon : float
        Maximum acceptable moon illumination. Must be between 0 and 1.

    minsep : interger or float
        Minimum acceptable angular separation between a target and the moon.

    minphase : float between 0 and 1
        Minimum acceptable moon phase for observation, as a fraction of moon
        phase with 1 = full moon and 0 = new moon.

    maxphase : float between 0 and 1
        Maximum acceptable moon phase for observation, as a fraction of moon
        phase with 1 = full moon and 0 = new moon.
    """

    def __init__(self , observer , times , altmin = 0 , altmax = 89 , airmass = 5 ,
               maxmoon = 0.9 , minsep = 15 , minphase = 0 , maxphase = 0.9):
        self.observer = observer
        self.times = times
        self.altmin = altmin
        self.altmax = altmax
        self.airmass = airmass
        self.maxmoon = maxmoon
        self.minsep = minsep
        self.maxphase = maxphase
        self.minphase = minphase

    """
    limits information:
    Takes altmin and altmax as degrees
    """

    def limits(self):
        altitude = AltitudeConstraint(self.altmin*u.deg , self.altmax*u.deg)
        air = AirmassConstraint(self.airmass)
        night = AtNightConstraint.twilight_astronomical()
        illumination = MoonIlluminationConstraint(max = self.maxmoon)
        sep = MoonSeparationConstraint(self.minsep*u.deg)
        phase = MoonPhaseConstraint(self.observer,
                                    self.times,
                                    self.minphase,
                                    self.maxphase)

        return [altitude , air , night , illumination , sep , phase]

    # def moonphase(self):
    #     return MoonPhaseConstraint(self.observer,
    #                                self.times,
    #                                self.minphase,
    #                                self.maxphase)


################################################################################
################################################################################


class observability():
    """
    This class will be used to perform calculations such as checking if targets
    will be visible within the specified time window for the observations,
    given constraints like target moon separation. The default constraint
    values are based on typical constraints for Veritas.
    """

#################################################################

    """
    Parameters
    ----------
    constraints : astroplan.constraint object or list of objects
        Defines constraints. Calculations to check observability of targets
        will be done using these AstroPlan constraint objects.
        For use with this class, the list of constraints defined in the
        limits class are ideal.

    observer : Astroplan Observer object
        Takes an AstroPlan observer object. These are returned by the Location
        class.

    targets : astroplan.FixedTarget object or list of objects
        An airmass of n requires the airmass be “better than n”.

    time : Time object
        For finding the moon phase, a time must be given to calculate the moon
        phase at that time.

    range : start and end time as Time objectsconstraints
        Returned by the Time_window.time_range() function defined previuosly.

    """


################## want to make a function that checks every day for a week
################## might be possible using datetime.datetime

#################################################################

    def __init__(self , constraints , observer , targets , range):
        self.constraints = constraints
        self.observer = observer
        self.targets = targets
        self.range = range

    def ever_observable(self):
        return is_observable(self.constraints,
                             self.observer,
                             self.targets,
                             time_range = Time(self.range))

    def when_observable(self):
        return observability_table(self.constraints,
                                 self.observer,
                                 self.targets,
                                 #self.range,
                                 time_range = Time(self.range),
                                 time_grid_resolution=100*u.second)

    def check(self):
        for j in range(np.size(self.targets)):
            print(self.targets[j].name)
            list1 = []
            list2 = []
            for i in range(np.size(self.constraints)):
                l = ("constraint" , i , is_observable(self.constraints[i],
                                                       self.observer,
                                                       self.targets[j],
                                                       time_range = Time(self.range)))
                list2.append(l)
            print(list2)
            list1.append(list2)
            list2 = []
        return list1


################################################################################
################################################################################


class schedule():

    """
    This class takes in time information, target information and constraint
    information to break the observation time window into observing blocks
    and assign a target to each block.
    """

#################################################################

    def __init__(self , exposure_time , readout_time , no_of_exposures,
                 exposure_time_per_target , targets , start_time,
                 end_time , constraints , transitioner , observation_blocks):
        self.exposure_time = exposure_time
        self.readout_time = readout_time
        self.no_of_exposures = no_of_exposures
        self.exposure_time_per_target = exposure_time_per_target
        self.targets = targets
        self.start_time = start_time
        self.end_time = end_time
        self.constraints = constraints
        self.transitioner = transitioner
        self.observation_blocks = observation_blocks


    """
    Parameters
    ----------
    constraints : astroplan.constraint object or list of objects
        Defines constraints. Calculations to check observability of targets
        will be done using these AstroPlan constraint objects.
        For use with this class, the list of constraints defined in the
        limits class are ideal.

    """

    """
    exposures_Default information:
    Sets the exposure time and number of exposures for each target to be equal
    at 10 exposures of 100 seconds each.
    """

    def exposures_default(exposure_time = 100 , readout_time = 20,
                          no_of_exposures = 10):
        return [exposure_time*u.second , readout_time*u.second ,
                no_of_exposures]

    """
    exposure_info information:
    Sets the exposure time for each individual target to be different but sets
    them with the same number of exposures. The exposure times should be given
    as a list.

    example:
        exp = [100 , 60 , 40 , 40 , 100 , 120 , 80]
        manual_exposure_info = cd.schedule.exposure_info(exp)
        print("manual exposure info = " , manual_exposure_info)
    """

    def exposure_info(exposure_time_per_target , readout_time = 20,
                      no_of_exposures = 10):
        times = []
        for i in range(np.size(exposure_time_per_target)):
            times.append(exposure_time_per_target[i]*u.second)

        return [times , readout_time*u.second , no_of_exposures]

    # """
    # default_observation_blocks information:
    # """
    #
    # def default_observation_blocks(start_time , end_time , targets):
    #     g

    """
    observation_blocks information:
    Takes the observation start and end times with the exposure information
    and breaks the observation time into time blocks. Takes a list of times
    to assign an exposure time per target.
    start_time and end_time take datetime.datetime objects, which are returned
    by the "range" function in the Time_window class.
    """
### the observation_blocks function takes a TimeConstraint separate
### to the global constraints on the rest of the observation
    def observation_blocks(targets , exposure_times,
                           readout_time , no_of_exposures,
                           start_time = startDefault , end_time = endDefault):
        #for priority, bandpass in enumerate(['B', 'G', 'R']):
        if np.size(targets) == 1:

            blocks = []
            night = TimeConstraint(Time(start_time) ,#- 5*u.minute,
                                   Time(end_time) )#+ 5*u.minute)
            print(targets)
            print(exposure_times)
            print(Time(start_time))
            print(Time(end_time))
            b = ObservingBlock.from_exposures(targets , #priority,
                                exposure_times, no_of_exposures,
                                readout_time,
                                #configuration = {'filter': bandpass},
                                constraints = night)
            blocks.append(b)

        else:

            blocks = []
            night = TimeConstraint(Time(start_time) , Time(end_time))
            for i in range(np.size(targets)):
                b = ObservingBlock.from_exposures(targets[i] , #priority,
                                    exposure_times[i], no_of_exposures,
                                    readout_time,
                                    #configuration = {'filter': bandpass},
                                    constraints = [night])
                print(b)
                blocks.append(b)
        return blocks

    """
    transitions information:
    Takes into account the time it takes the observatory to slew between
    targets. Can later include information about time needed to transition
    between different filters.
    """

    def transitions(slew_rate = 1):
        return Transitioner(slew_rate*u.deg/u.second)

    """
    schedule information:
    Takes the initialized sheduling and block information and constructs
    an observation schedule from the initialized information.
    Uses the AstroPlan SequentialScheduler class.
    The class "Schedule" with a capital "S" is from AstroPlan, "schedule" with
    a lower case "s" is from this class.
    """

    def schedule(constraints , transitioner , observation_blocks,
                 observer = VeritasLocationDefault,
                 #start = '2019-07-15 20:00:00' , end = '2019-07-16 04:30:00'):
                 start = startDefault , end = endDefault):
        ### Initialize the sequential scheduler with the global constraints
        ### and transitioner (with slew rate and filter change info)
        print(np.size(observation_blocks))
        seq_scheduler = SequentialScheduler(constraints,
                                            observer,
                                            transitioner)

        ### Initialize a Schedule object, to contain the new schedule
        sequential_schedule = Schedule(Time(start) , Time(end))

        ### Call the schedule with the observing blocks and schedule to
        ### schedule the blocks
        plt.figure()#figsize = (14,6))
        plot_schedule_airmass(sequential_schedule)
        plt.tight_layout()
        plt.legend(loc="upper right")
        plt.show()

        return seq_scheduler(observation_blocks, sequential_schedule).to_table()

    def block(targets, start_time = eveningDublin, end_time = morningDublin):
        blocks =[]

        if np.size(targets) == 1:
            blocks.append(ObservingBlock(targets , 1*u.hour , 0))
            print("blocks size = " , np.size(blocks))
            transitioner = Transitioner(slew_rate=2*u.deg/u.second)
            schedule = Schedule(Time(start_time) , Time(end_time))
            observer = Dublin
            scheduler = SequentialScheduler(constraints = [],
                                                observer = observer,
                                                transitioner = transitioner)
        else:
            for i in range(np.size(targets)):
                blocks.append( ObservingBlock(targets[i] , 30*u.minute , 0) )
                print("blocks size = " , np.size(blocks))
            transitioner = Transitioner(slew_rate=2*u.deg/u.second)
            schedule = Schedule(Time(start_time) , Time(end_time))
            observer = Dublin
            scheduler = SequentialScheduler(constraints = [],
                                            observer = observer,
                                            transitioner = transitioner)

        # print(scheduler(blocks, schedule))
        return tabulate( [scheduler(blocks, schedule).to_table()] )
        # (plot_schedule_airmass(scheduler),
        #         plt.legend(loc = 'best'),
        #         plt.show() )


    def year_check(constraints , observer , targets , time_range):
        time = Time(time_range)
        dates = []
        for i in range(364):
            table = observability_table(constraints,
                                     observer,
                                     targets,
                                     #self.range,
                                     time_range = time_range,
                                     time_grid_resolution=100*u.second)
            fraction_of_time_observable = table[i][3] *u.d

            blocks = []

            blocks.append(ObservingBlock(targets ,
                                         fraction_of_time_observable , 0))
            # print("blocks size = " , np.size(blocks))
            transitioner = Transitioner(slew_rate=1.2*u.deg/u.second)
            schedule = Schedule(Time(time_range[0]) , Time(time_range[1]))
            observer = Dublin
            scheduler = SequentialScheduler(constraints = [],
                                                observer = observer,
                                                transitioner = transitioner)
            x = Time(time_range[0]) + 1*u.d
            y = Time(time_range[1]) + 1*u.d
            time = (x , y)

            dates.append( scheduler(blocks, schedule) )#.to_table() )

        for i in range(np.size(dates)):
            tabulate = tabulate( [dates[i].to_table()] )

        return tabulate






############################## maybe start making a new function that can check
############################## times visible for a target for a given date


################################################################################
