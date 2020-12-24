#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 14:09:21 2020

@author: jacobfaibussowitsch
"""
import os

petscDir = os.getenv("PETSC_DIR")
imDir = os.path.join(petscDir, "src", "docs", "website", "images")
if not os.path.exists(imDir):
    raise RuntimeError("Image directory "+imDir+" has moved or been deleted!")

ownerURL = "https://gitlab.com/api/v4/groups/petsc/members/all"
integratorURL = "https://gitlab.com/api/v4/groups/5583565/members/all"
devURL = "https://gitlab.com/api/v4/groups/5981367/members/all"

emeritus = {
    "william_gropp" : {
        "web_url" : "https://cs.illinois.edu/directory/profile/wgropp",
        "avatar_url" : os.path.join(imDir, "bill.gif"),
        "name" : "William Gropp"
    },
    "victor_eijkhout" : {
        "web_url" : "https://www.tacc.utexas.edu/staff/victor-eijkhout",
        "avatar_url" : os.path.join(imDir, "victor.jpg"),
        "name" : "Victor Eijkhout"
    },
    "peter_brune" : {
        "web_url" : "",
        "avatar_url" : os.path.join(imDir, "peter.jpg"),
        "name" : "Peter Brune"
    },
    "kris_buschelman" : {
        "web_url" : "",
        "avatar_url" : os.path.join(imDir, "buschelman.jpg"),
        "name" : "Kris Buschelman"
    },
    "sean_farley" : {
        "web_url" : "https://farley.io/",
        "avatar_url" : os.path.join(imDir, "sean.jpg"),
        "name" : "Sean Farley"
    },
    "dmitry_karpeev" : {
        "web_url" : "https://www.ci.uchicago.edu/profile/224",
        "avatar_url" : os.path.join(imDir, "dmitry.jpg"),
        "name" : "Dmitry Karpeev"
    },
    "dinesh_kaushik" : {
        "web_url" : "",
        "avatar_url" : os.path.join(imDir, "dinesh.jpg"),
        "name" : "Dinesh Kaushik"
    },
    "jason_sarich" : {
        "web_url" : "https://www.anl.gov/mcs/person/jason-sarich",
        "avatar_url" : os.path.join(imDir, "sarich.jpg"),
        "name" : "Jason Sarich"
    },
    "victor_minden" : {
        "web_url" : "",
        "avatar_url" : os.path.join(imDir, "victorminden.jpg"),
        "name" : "Victor Minden"
    }
}

activeCoreDevs = {
    "lois.curfman.mcinnes" : {
        "web_url" : "https://press3.mcs.anl.gov/curfman/",
        "avatar_url" : os.path.join(imDir, "lois.gif"),
        "name" : "Lois Curfman Mcinnes"
    },
    "sbalay" : {
        "web_url" : None,
        "avatar_url" : None,
        "name" : None
    },
    "jedbrown" : {
        "web_url" : None,
        "avatar_url" : None,
        "name" : None
    },
    "adener" : {
        "web_url" : None,
        "avatar_url" : None,
        "name" : None
    },
    "blaisebourdin" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "dmay" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "fdkong" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "ghammond" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "hannah_mairs" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "hongzhangsun" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "jfaibussowitsch" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "caidao22" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "jczhang07" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "karlrupp" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "markadams4" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : "Mark Adams"
    },
    "knepley" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "oanam198" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "psanan" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "rtmills" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "abhyshr" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "stefanozampini" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "tmunson" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "haplav" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "prj-" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : "P RJ"
    },
    "wence" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "tisaac" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "krugers" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "dalcinl" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "joseroman" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "bwhitchurch" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    },
    "barrysmith" : {
	"web_url" : None,
	"avatar_url" : None,
	"name" : None
    }
}

def getJson(url, token):
    import requests

    headers = {"PRIVATE-TOKEN" : token}
    rpage = requests.get(url, headers = headers)
    rpage.raise_for_status()
    numPages = int(rpage.headers["X-Total-Pages"])
    lst = []
    for i in range(numPages):
        r = requests.get(url, headers = headers, params = {"page" : i+1})
        r.raise_for_status()
        lst.extend(r.json())
    return lst

def makeDevDict(devJson):
    # for dev in devJson:
    #     doubleTrouble = False
    #     alevel = dev["access_level"]
    #     if alevel not in tierDict:
    #         tierDict[alevel] = {}
    #     # Need to re-traverse entire dict to check if people are double counted, since
    #     # some people have themselves under multiple roles...
    #     for level in tierDict:
    #         if dev["name"] in tierDict[level]:
    #             # boil and bubble...
    #             doubleTrouble = True
    #             break
    #     if not doubleTrouble:
    #         tierDict[alevel][dev["name"]] = [dev["web_url"], dev["avatar_url"]]
    # Add list of extra devs to dict
    for dev in devJson:
        ldev = dev["username"].lower()
        if ldev in activeCoreDevs:
            for key, val in activeCoreDevs[ldev].items():
                if val is None:
                    activeCoreDevs[ldev][key] = dev[key]
    return activeCoreDevs

def writeRst(fname, devs):
    with open(fname, "w+") as f:
        f.writelines([
            ".. raw:: html\n\n",
            "   <!-- Generated by generate_authors_table.py -->\n",
            "   <div class=\"petsc-team-container\">\n",
            "   <style>\n",
            "     img.avatar {border-radius: 10px;width: 60px;height: 60px;}\n",
            "   </style>\n"
        ])
        lines = []
        for dev in sorted(devs.items(), key = lambda item: item[1]["name"].split(" ")[-1]):
            lines.append("    <div>\n")
            lines.append("    <a href='%s'><img src='%s' class='avatar' /></a> <br />\n" %
                         (dev[1]["web_url"], dev[1]["avatar_url"]))
            lines.append("    <p>%s</p>\n" % (dev[1]["name"]))
            lines.append("    </div>\n")
        lines.append("    </div>\n")
        f.writelines(lines)
    print("Wrote table to "+fname)

if __name__ == "__main__":
    import argparse
    import pathlib

    string = "<string>"
    path = "<path/to/dir>"
    parser = argparse.ArgumentParser(description = "Build Developer Table", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--gitlab-token", required = True, metavar = string, help = "Specify your private Gitlab Authentican Token", dest = "token")
    parser.add_argument("-o", "--output-dir", required = True, metavar = path, type = pathlib.Path, help = "Specify the output directory", dest = "writeDir")
    args = parser.parse_args()
    devJson = getJson(devURL, args.token)
    ownerJson = getJson(ownerURL, args.token)
    integratorJson = getJson(integratorURL, args.token)
    megaJson = devJson+ownerJson+integratorJson
    tierlist = makeDevDict(megaJson)
    writeDir = args.writeDir
    if not os.path.exists(writeDir):
        os.makedirs(writeDir)
    currentFile = os.path.join(writeDir, "petsc-team-table.inc")
    writeRst(currentFile, tierlist)
    emeritusFile = os.path.join(writeDir, "petsc-emeritus-table.inc")
    writeRst(emeritusFile, emeritus)
