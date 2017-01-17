import xmltodict
from bunch import *
from xml.etree.ElementTree import Element, tostring


def xml_to_dict(el):
  d={}
  if el.text:
    d[el.tag] = el.text
  else:
    d[el.tag] = {}
  children = el.getchildren()
  if children:
    d[el.tag] = map(xml_to_dict, children)
  return d


def bxml(doc):

	with open(doc) as fd:
    		obj = xmltodict.parse(fd.read())

	return bunchify(obj)



def dict_to_xml(tag, d):
    '''
    Turn a simple dict of key/value pairs into XML
    '''
    elem = Element(tag)
    for key, val in d.items():
        child = Element(key)
        child.text = str(val)
        elem.append(child)
    return tostring(elem)



