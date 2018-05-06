
# coding: utf-8
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
from sgp4.io import twoline2rv
from sgp4.ext import jday
import datetime
from operator import attrgetter
import math
import numpy as np
import pandas as pd
import copy


class SpaceObjects(object):
    """Space objects to propagate in pararell by SGP4 using tensorflow.

    Attributes:
        satrecs (list of :obj:`Satellite`): TLE Sattelite objects list.
        date_list (list of :obj:`datetime`): Propagate date in UTC.
        jdays (list of str): Propagate date in julian day.
        split_list (list of int):Categorized number according to orbital characteristics.
    """

    def __init__(self, tle_list, whichconst, start_epoch, propagate_range, delta_t=60, julian_date=False,
                 tf_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)):
        self.satrecs = self.read_twolinelist(tle_list, whichconst)
        self.date_list, self.jdays = self.__make_jday_list(start_epoch, propagate_range, delta_t, julian_date)
        #  ------------------ make tensor split_list -------------------
        num_of_deepspace = len([satrec for satrec in self.satrecs if satrec.method == 'd' and satrec.isimp == 1])
        num_of_normal = len([satrec for satrec in self.satrecs if satrec.isimp == 0])
        num_of_under220km = len([satrec for satrec in self.satrecs if satrec.method == 'n' and satrec.isimp == 1])
        self.split_list = [num_of_deepspace, num_of_normal, num_of_under220km]
        self.tf_config=tf_config


    def __make_plenty_of_tensor_lists(self, *args):
        """Wrapper for __make_tensor_list().

        It depents on self.__make_tensor_list and self.satrecs.

        Args:
            *args (str lists): Attribute names to hand over __make_tensor_list().

        Returns:
            tuple: multiple tensor_lists for each specified attribute.
        """
        tensor_lists = []
        for arg in args:
            tensor_lists.append(self.__make_tensor_list(arg, self.satrecs))

        return tuple(tensor_lists)

    def __make_tensor_list(self, attr, obj_list, dtype='float64'):
        """Make tensor list from 'split_list'.

        Create tensor from a list of Satellite object and
        one of its 'Attribute' for each 'jdays'.
        It depents on __make_tensor_list and satrecs.

        Args:
            attr (str): Attribute names for hand over __make_tensor_list().
            obj_list (list of :obj:`Satellite`)
            dtype (str): Data type of tensor element.

        Returns:
            tuple: multiple tensor_lists for each specified attribute.
        """
        try:
            tensor = tf.transpose(tf.constant([[getattr(obj, attr) for obj in obj_list] for x in self.jdays], dtype='float64'))
            tensor_list = tf.split(tensor, self.split_list, 0)
            return tensor_list
        except AttributeError:
            print("invalid argment")
            raise

    def __make_jday_list(self, start_epoch=datetime.datetime.today(), propagate_range=10, delta_t=1, julian_date=False):
        """Make datetime and julian day list.

        Args:
            start_epoch (:obj:datetime): Epoch when SGP4 propagation start.
            propagate_range (int): SGP4 propagation period in seconds.
            telta_t (int): SGP4 propagation interval in seconds.

        Returns:
            tuple: datetime list and julian day list.
        """
        if julian_date == True:
            date_list = [x for x in self.__drange(start_epoch, propagate_range, delta_t)]
            jdays = date_list
        else:
            # make datetime list
            date_list = [start_epoch + datetime.timedelta(seconds=x) for x in range(0, propagate_range, delta_t)]
            # make julian day list using jday function in sgp4.ext
            jdays = [jday(j.year, j.month, j.day, j.hour, j.minute, j.second) for j in date_list]

        return date_list, jdays

    def __drange(self, begin, end, step):
        """Extention of range() function that accept real number.
        """
        n = begin
        while n+step <= end+step:
            yield n
            n += step

    def __make_gravitational_constants(self):
        """Make gravitational constatnts.
        """
        whichconsts = [satrec.whichconst for satrec in self.satrecs]

        tumin = self.__make_tensor_list('tumin', whichconsts)
        mu = self.__make_tensor_list('mu', whichconsts)
        radiusearthkm = self.__make_tensor_list('radiusearthkm', whichconsts)
        xke = self.__make_tensor_list('xke', whichconsts)
        j2 = self.__make_tensor_list('j2', whichconsts)
        j3 = self.__make_tensor_list('j3', whichconsts)
        j4 = self.__make_tensor_list('j4', whichconsts)
        j3oj2 = self.__make_tensor_list('j3oj2', whichconsts)

        return tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2

    def __make_orbit_element_dataframe(self, position, velocity):
        """Make Pandas DataFrame from position and velocity ndarray.

        Args:
            position (nadarray): Position state vectors.
            velocity (nadarray): Velocity state vectors.

        Returns:
            Pandas DataFrame: State vectors of each time and each satnum.
        """
        x, y, z = position
        vx, vy, vz = velocity

        satnums = [satrec.satnum for satrec in self.satrecs]
        name_x = pd.DataFrame(np.vstack((satnums, ['x' for i in satnums])).T, columns=['satnum', 'element'])
        df_x = pd.DataFrame(x, columns=self.date_list)
        df_x = pd.concat([name_x, df_x], axis=1)

        name_y = pd.DataFrame(np.vstack((satnums, ['y' for i in satnums])).T, columns=['satnum', 'element'])
        df_y = pd.DataFrame(y, columns=self.date_list)
        df_y = pd.concat([name_y, df_y], axis=1)

        name_z = pd.DataFrame(np.vstack((satnums, ['z' for i in satnums])).T, columns=['satnum', 'element'])
        df_z = pd.DataFrame(z, columns=self.date_list)
        df_z = pd.concat([name_z, df_z], axis=1)

        name_vx = pd.DataFrame(np.vstack((satnums, ['vx' for i in satnums])).T, columns=['satnum', 'element'])
        df_vx = pd.DataFrame(vx, columns=self.date_list)
        df_vx = pd.concat([name_vx, df_vx], axis=1)

        name_vy = pd.DataFrame(np.vstack((satnums, ['vy' for i in satnums])).T, columns=['satnum', 'element'])
        df_vy = pd.DataFrame(vy, columns=self.date_list)
        df_vy = pd.concat([name_vy, df_vy], axis=1)

        name_vz = pd.DataFrame(np.vstack((satnums, ['vz' for i in satnums])).T, columns=['satnum', 'element'])
        df_vz = pd.DataFrame(vz, columns=self.date_list)
        df_vz = pd.concat([name_vz, df_vz], axis=1)

        df_r = pd.concat([df_x, df_y, df_z, df_vx, df_vy, df_vz], axis=0)
        df_r.satnum = df_r.satnum.astype(np.int64)
        df_r = df_r.set_index(['satnum', 'element'], drop=True)

        return df_r

    def __replace_tensor_elem_with_condition(self, tensor, condition, value, replace):
        """Replace tensor elements which match a condition.

        Args:
            tensor (tensor): Target tensor to replace element.
            condition (str): Conditioal string "less" or "greater" or "equal".
            value (float): Comparing value for each tensor element.
            replace (float): Replacement value for which match the condition.

        Returns:
            tensor: modified tensor.
        """
        if condition == 'less':
            cond = tf.less(tensor, value)
        elif condition == 'greater':
            cond = tf.greater(tensor, value)
        elif condition == 'equal':
            cond = tf.equal(tensor, value)

        case_true = tf.multiply(tf.ones(tf.shape(tensor), tf.float64), replace)
        case_false = tensor
        tensor_m = tf.where(cond, case_true, case_false)
        return tensor_m

    def read_twolinelist(self, twolinelist, whichconst):
        """Read several twolines at once.

        Args:
            twolinelist (list of str): TLE list.
            whichconst (:obj:`EarthGravity`): Earth gravitational constant.

        Returns:
            list of Satellite: Satelite object list initialized by sgp4_init.
        """
        args1 = []
        args2 = []
        args3 = []
        for n in range(0, len(twolinelist)-1, 2):
            args1.append(twolinelist[n:n+2][0])
            args2.append(twolinelist[n:n+2][1])
            args3.append(whichconst)

        with ThreadPoolExecutor() as executor:
            satrecs = list(executor.map(twoline2rv, args1, args2, args3))

        return satrecs

    def thread_pararell_sgp4_check(self):
        """Check the speed of concurrent thread execution.
        """
        with ThreadPoolExecutor() as executor:
           for sat in self.satrecs:
               list(executor.map(sat.propagate,
                                 [i.year for i in self.date_list],
                                 [i.month for i in self.date_list],
                                 [i.day for i in self.date_list],
                                 [i.hour for i in self.date_list],
                                 [i.minute for i in self.date_list],
                                 [i.second for i in self.date_list]))


    def propagate(self):
        """Multiple SGP4 propagation for TLE list.

        Returns:
            Pandas DataFrame: State vectors of each time and each satnum.
        """
        #  ------------------ set mathematical constants ---------------
        twopi = 2.0 * math.pi
        x2o3 = 2.0 / 3.0
        #  --------------------- clear sgp4 error flag -----------------
        for satrec in self.satrecs:
            satrec.t = self.jdays
            satrec.error = 0
            satrec.error_message = None

        # sort the satrecs (satrec list) by 'method' and 'isimp'
        # index 0: (method='d', isimp=1) (Deep Space）
        # index 1: (method='n', isimp=0) (SGP4)
        # index 2: (method='n', isimp=1) (SGP4: perigee altitude less than 220 km)
        self.satrecs = sorted(self.satrecs, key=attrgetter('method', 'isimp'))

        # make gravitational constants
        tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2 = \
            self.__make_gravitational_constants()

        #  -------------------- initialize list valuables ------------------
        vkmpersec = []
        xmdf = []
        argpdf = []
        nodedf = []
        t2 = []
        nodem = []
        tempa = []
        tempe = []
        templ = []
        am = []
        temp1 = []
        xlm = []
        emsq = []
        sinim = []
        cosim = []
        ep = []
        xincp = []
        argpp = []
        nodep = []
        mp = []
        sinip = []
        cosip = []
        axnl = []
        temp3 = []
        aynl = []
        xl = []
        u = []
        eo1 = []
        tem5 = []
        sineo1 = []
        coseo1 = []
        ecose = []
        esine = []
        el2 = []
        pl = []
        rl = []
        rdotl = []
        rvdotl = []
        betal = []
        temp4 = []
        sinu = []
        cosu = []
        su = []
        sin2u = []
        cos2u = []
        temp5 = []
        temp6 = []
        temp7 = []
        mrt = []
        xnode = []
        xinc = []
        mvt = []
        rvdot = []
        sinsu = []
        cossu = []
        snod = []
        cnod = []
        sini = []
        cosi = []
        xmx = []
        xmy = []
        ux = []
        uy = []
        uz = []
        vx = []
        vy = []
        vz = []
        _mr = []
        r = []
        v = []

        #  -------------------- initialize tensor valuables ------------------
        t = tf.constant([satrec.t for satrec in self.satrecs], dtype='float64')
        t = tf.split(t, self.split_list, 0)

        mo, mdot, argpo, argpdot, nodeo, nodedot, nodecf, cc1, bstar, cc4, \
        t2cof, omgcof, eta, xmcof, delmo, d2, d3, d4, sinmao, cc5, t3cof, \
        t4cof, t5cof, no, ecco, inclo, aycof, xlcof, con41, x1mth2, x7thm1 =\
        self.__make_plenty_of_tensor_lists(
            'mo', 'mdot', 'argpo', 'argpdot', 'nodeo', 'nodedot', 'nodecf', 'cc1', 'bstar', 'cc4',
            't2cof', 'omgcof', 'eta', 'xmcof', 'delmo', 'd2', 'd3', 'd4', 'sinmao', 'cc5', 't3cof',
            't4cof', 't5cof', 'no', 'ecco', 'inclo', 'aycof', 'xlcof', 'con41', 'x1mth2', 'x7thm1')

        for i in range(3):
            vkmpersec.append(radiusearthkm[i] * xke[i]/60.0)

            #  ------- update for secular gravity and atmospheric drag -----
            xmdf.append(mo[i] + mdot[i] * t[i])
            argpdf.append(argpo[i] + argpdot[i] * t[i])
            nodedf.append(nodeo[i] + nodedot[i] * t[i])
            t2.append(t[i] * t[i])
            nodem.append(nodedf[i] + nodecf[i] * t2[i])
            tempa.append(1.0 - cc1[i] * t[i])
            tempe.append(bstar[i] * cc4[i] * t[i])
            templ.append(t2cof[i] * t2[i])

        argpm = copy.copy(argpdf)
        mm = copy.copy(xmdf)

        # only for index 1: (method='n', isimp=0) (LEO)
        delomg = omgcof[1] * t[1]
        delmtemp = 1.0 + eta[1] * tf.cos(xmdf[1])
        delm = xmcof[1] * (tf.pow(delmtemp, 3) - delmo[1])
        temp0 = delomg + delm

        mm[1] = xmdf[1] + temp0
        argpm[1] = argpdf[1] - temp0

        t3 = t2[1] * t[1]
        t4 = t3 * t[1]

        tempa[1] = tempa[1] - d2[1] * t2[1] - d3[1] * t3 - d4[1] * t4
        tempe[1] = tempe[1] + bstar[1] * cc5[1] * (tf.sin(mm[1]) - sinmao[1])
        templ[1] = templ[1] + t3cof[1] * t3 + t4 * (t4cof[1] + t[1] * t5cof[1])

        nm = copy.copy(no)
        em = copy.copy(ecco)
        inclm = copy.copy(inclo)

        # TODO: implement SDP (Deep Space Object propagation)

        for i in range(3):
            am.append(tf.pow((xke[i] / nm[i]), x2o3) * tempa[i] * tempa[i])
            nm[i] = xke[i] / tf.pow(am[i], 1.5)
            em[i] = em[i] - tempe[i]
            em[i] = self.__replace_tensor_elem_with_condition(em[i], 'less', 1e-6, 1e-6)
            mm[i] = mm[i] + no[i] * templ[i]
            xlm.append(mm[i] + argpm[i] + nodem[i])
            emsq.append(em[i] * em[i])
            temp1.append(1.0 - emsq[i])
            nodem[i] = tf.mod(nodem[i], twopi)
            argpm[i] = tf.mod(argpm[i], twopi)
            xlm[i] = tf.mod(xlm[i], twopi)
            mm[i] = tf.mod((xlm[i] - argpm[i] - nodem[i]), twopi)
            #  ----------------- compute extra mean quantities -------------
            sinim.append(tf.sin(inclm[i]))
            cosim.append(tf.cos(inclm[i]))
            #  -------------------- add lunar-solar periodics --------------
            ep.append(em[i])
            xincp.append(inclm[i])
            argpp.append(argpm[i])
            nodep.append(nodem[i])
            mp.append(mm[i])
            sinip.append(sinim[i])
            cosip.append(cosim[i])

            # TODO: implement SDP (Deep Space Object propagation)

            axnl.append(ep[i] * tf.cos(argpp[i]))
            temp3.append(1.0 / (am[i] * (1.0 - ep[i] * ep[i])))
            aynl.append(ep[i] * tf.sin(argpp[i]) + temp3[i] * aycof[i])
            xl.append(mp[i] + argpp[i] + nodep[i] + temp3[i] * xlcof[i] * axnl[i])

            #  --------------------- solve kepler's equation ---------------
            u.append(tf.mod((xl[i] - nodep[i]), twopi))
            eo1.append(u[i])
            sineo1.append(tf.sin(eo1[i]))
            coseo1.append(tf.cos(eo1[i]))
            tem5.append(9999.9)
            #    sgp4fix for kepler iteration
            #    the following iteration needs better limits on corrections
            for ktr in range(10):
                sineo1[i] = tf.sin(eo1[i])
                coseo1[i] = tf.cos(eo1[i])
                tem5[i] = 1.0 - coseo1[i] * axnl[i] - sineo1[i] * aynl[i]
                tem5[i] = (u[i] - aynl[i] * coseo1[i] + axnl[i] * sineo1[i] - eo1[i]) / tem5[i]
                eo1[i] = eo1[i] + tem5[i]

            #  ------------- short period preliminary quantities -----------

            ecose.append(axnl[i]*coseo1[i] + aynl[i]*sineo1[i])
            esine.append(axnl[i]*sineo1[i] - aynl[i]*coseo1[i])
            el2.append(axnl[i]*axnl[i] + aynl[i]*aynl[i])
            pl.append(am[i]*(1.0-el2[i]))

            rl.append(am[i] * (1.0 - ecose[i]))
            rdotl.append(tf.sqrt(am[i]) * esine[i]/rl[i])
            rvdotl.append(tf.sqrt(pl[i]) / rl[i])
            betal.append(tf.sqrt(1.0 - el2[i]))
            temp4.append(esine[i] / (1.0 + betal[i]))
            sinu.append(am[i] / rl[i] * (sineo1[i] - aynl[i] - axnl[i] * temp4[i]))
            cosu.append(am[i] / rl[i] * (coseo1[i] - axnl[i] + aynl[i] * temp4[i]))
            su.append(tf.atan2(sinu[i], cosu[i]))
            sin2u.append((cosu[i] + cosu[i]) * sinu[i])
            cos2u.append(1.0 - 2.0 * sinu[i] * sinu[i])
            temp5.append(1.0 / pl[i])
            temp6.append(0.5 * j2[i] * temp5[i])
            temp7.append(temp6[i] * temp5[i])

            #  -------------- update for short period periodics ------------
            # TODO: implement SDP (Deep Space Object propagation)
            mrt.append(rl[i] * (1.0 - 1.5 * temp7[i] * betal[i] * con41[i]) + 0.5 * temp6[i] * x1mth2[i] * cos2u[i])
            su[i] = su[i] - 0.25 * temp7[i] * x7thm1[i] * sin2u[i]
            xnode.append(nodep[i] + 1.5 * temp7[i] * cosip[i] * sin2u[i])
            xinc.append(xincp[i] + 1.5 * temp7[i] * cosip[i] * sinip[i] * cos2u[i])
            mvt.append(rdotl[i] - nm[i] * temp6[i] * x1mth2[i] * sin2u[i] / xke[i])
            rvdot.append(rvdotl[i] + nm[i] * temp6[i] * (x1mth2[i] * cos2u[i] + 1.5 * con41[i]) / xke[i])

            #  --------------------- orientation vectors -------------------
            sinsu.append(tf.sin(su[i]))
            cossu.append(tf.cos(su[i]))
            snod.append(tf.sin(xnode[i]))
            cnod.append(tf.cos(xnode[i]))
            sini.append(tf.sin(xinc[i]))
            cosi.append(tf.cos(xinc[i]))
            xmx.append(-snod[i] * cosi[i])
            xmy.append(cnod[i] * cosi[i])
            ux.append(xmx[i] * sinsu[i] + cnod[i] * cossu[i])
            uy.append(xmy[i] * sinsu[i] + snod[i] * cossu[i])
            uz.append(sini[i] * sinsu[i])
            vx.append(xmx[i] * cossu[i] - cnod[i] * sinsu[i])
            vy.append(xmy[i] * cossu[i] - snod[i] * sinsu[i])
            vz.append(sini[i] * cossu[i])

            #  --------- position and velocity (in km and km/sec) ----------
            _mr.append(mrt[i] * radiusearthkm[i])
            r.append((_mr[i] * ux[i], _mr[i] * uy[i], _mr[i] * uz[i]))
            v.append(((mvt[i] * ux[i] + rvdot[i] * vx[i]) * vkmpersec[i],
                     (mvt[i] * uy[i] + rvdot[i] * vy[i]) * vkmpersec[i],
                     (mvt[i] * uz[i] + rvdot[i] * vz[i]) * vkmpersec[i]))

        # index 0: (method='d', isimp=1) (Deep Space）
        # index 1: (method='n', isimp=0) (SGP4)
        # index 2: (method='n', isimp=1) (SGP4: perigee altitude less than 220 km)
        with tf.Session(config=self.tf_config) as sess:
            r0, r1, r2 = sess.run(r)
            v0, v1, v2 = sess.run(v)

        position = np.hstack([r0, r1, r2])
        velocity = np.hstack([v0, v1, v2])

        df_r = self.__make_orbit_element_dataframe(position, velocity)

        return df_r
