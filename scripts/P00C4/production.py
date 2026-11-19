"""Frozen C3 physics and solver; add the same required outer-gradient outputs as proxy."""

import json
import numpy as np
from common import *
from grid import build

SIZE = {"coarse": (0.8, 0.010, 0.20), "medium": (0.5, 0.006, 0.13), "fine": (0.3, 0.003, 0.085)}


def parameters(d, k):
    cmd = []
    for r in read(ROOT / "reports/P00C1/PARAMETERS.csv"):
        val = str(d) if r["name"] == "d" else str(k) if r["name"] == "perm" else r["value"]
        if not val:
            continue
        if r["origin_type"] == "DERIVED_PHYSICS":
            val = r["expression"]
        elif r["unit"] != "1":
            val += "[" + r["unit"] + "]"
        cmd.append(f'm.param().set("{r["name"]}","{val}");')
    return "\n".join(cmd)


def extra(coords, outer, archive=True):
    s = r"""
 double[] engineering={3600,21600,86400,259200,604800,864000,1728000,2592000,5184000,7776000};
 var ex=m.result().export().create("fields","Data");ex.set("data","dset1");ex.set("location","file");ex.set("coordfilename","__COORDS__");
 ex.set("expr",new String[]{"pmAbs","p","SmGas","SfGas","GammaX"});ex.set("header","on");ex.set("fullprec","on");ex.set("separator",",");ex.set("sort","off");ex.set("solrepresentation","solnum");ex.set("innerinput","manual");
 for(double target:engineering){int n=-1;for(int j=0;j<tt.length;j++)if(Math.abs(tt[j]-target)<1e-8)n=j+1;if(n<1)throw new Exception("no snapshot");ex.set("solnum",new int[]{n});ex.set("filename",out+"field_"+Math.round(target)+".csv");ex.run();}
 mark("F_exported");
 var ox=m.result().export().create("outertest","Data");ox.set("data","dset1");ox.set("location","file");ox.set("coordfilename","__OUTER__");ox.set("expr",new String[]{"p","d(p,x)","d(p,y)"});ox.set("header","on");ox.set("fullprec","on");ox.set("separator",",");ox.set("sort","off");ox.set("solrepresentation","solnum");ox.set("innerinput","manual");
 for(double target:engineering){int n=-1;for(int j=0;j<tt.length;j++)if(Math.abs(tt[j]-target)<1e-8)n=j+1;ox.set("solnum",new int[]{n});ox.set("filename",out+"outer_"+Math.round(target)+".csv");ox.run();}
 var wall=m.result().numerical().create("wallmax","MaxLine");wall.selection().named("well1");wall.set("expr","abs(p-pw)");double[] wa=wall.getReal()[0];wall.selection().named("well2");double[] wb=wall.getReal()[0];double[][] wr=new double[tt.length][3];for(int j=0;j<tt.length;j++)wr[j]=new double[]{tt[j],wa[j],wb[j]};table("wall","t_s,wall1_Pa,wall2_Pa",wr);
 mark("Fplus_exported");
 __ARCHIVE__
 mark("complete");ModelUtil.remove(m.tag());
 """.replace(
        "__COORDS__", coords.as_posix()
    ).replace(
        "__OUTER__", outer.as_posix()
    )
    archive_code = r"""
 java.util.Set<Integer> keep=new java.util.HashSet<>();for(double target:new double[]{0,.01,3600,21600,86400,259200,604800,864000,1728000,2592000,5184000,7776000})for(int j=0;j<tt.length;j++)if(Math.abs(tt[j]-target)<1e-8)keep.add(j+1);
 int[] drop=new int[tt.length-keep.size()];int z=0;for(int j=1;j<=tt.length;j++)if(!keep.contains(j))drop[z++]=j;
 var trim=m.sol("sol1").create("trim","CombineSolution");trim.set("soloper","remsol");trim.set("removesol","current");trim.set("excludeorinclude","exclude");trim.set("excmethod","explicit");trim.set("solnum","manual");trim.set("manualsolnum",drop);m.sol("sol1").runFromTo("trim","trim");m.sol("sol1").feature().remove("trim");m.save(out+"review.mph");
 """
    return s.replace("__ARCHIVE__", archive_code if archive else "")
