import xml.etree.ElementTree as ET

def writeXml(progress: float = 0, prompt: str = '', neg_prompt: str = '', message: str = '') -> bytes:
    xmlMetaData = ET.Element('TouchDesignerFormat')
    ET.SubElement(xmlMetaData, "HeaderInfo", version="1", datatype="table", numrows="2", numcolumns="3")
    data: ET.Element = ET.SubElement(xmlMetaData, "Data")

    row0: ET.Element = ET.SubElement(data, "Row", row="0")
    column0 = ET.SubElement(row0, "Column", column="0")
    ET.SubElement(column0, 'Cell', value = 'progress')
    column1 = ET.SubElement(row0, "Column", column="1")
    ET.SubElement(column1, 'Cell', value = 'prompt')
    column1 = ET.SubElement(row0, "Column", column="2")
    ET.SubElement(column1, 'Cell', value = 'negative prompt')
    column1 = ET.SubElement(row0, "Column", column="3")
    ET.SubElement(column1, 'Cell', value = 'message')

    row1: ET.Element = ET.SubElement(data, "Row", row="1")
    column0: ET.Element = ET.SubElement(row1, "Column", column="0")
    ET.SubElement(column0, 'Cell', value = str(progress))
    column1 = ET.SubElement(row1, "Column", column="1")
    ET.SubElement(column1, 'Cell', value = prompt)
    column1 = ET.SubElement(row1, "Column", column="2")
    ET.SubElement(column1, 'Cell', value = neg_prompt)
    column1: ET.Element = ET.SubElement(row1, "Column", column="3")
    ET.SubElement(column1, 'Cell', value = message)

    return ET.tostring(xmlMetaData, method='xml')

def readXml(xmlString: bytes | str) -> tuple[float, str, str, str]:
    xmlElement: ET.Element = ET.fromstring(xmlString)

    progress_elem: ET.Element | None = xmlElement.find(".//Data/Row[@row='1']/Column[@column='0']/Cell")
    progress_value: str | None = progress_elem.get('value') if progress_elem is not None else None
    progress: float = float(progress_value) if progress_value else 0.0

    prompt_elem: ET.Element | None = xmlElement.find(".//Data/Row[@row='1']/Column[@column='1']/Cell")
    prompt_value: str | None = prompt_elem.get('value') if prompt_elem is not None else None
    prompt: str = prompt_value if prompt_value is not None else ''

    neg_elem: ET.Element | None = xmlElement.find(".//Data/Row[@row='1']/Column[@column='2']/Cell")
    neg_value: str | None = neg_elem.get('value') if neg_elem is not None else None
    neg_prompt: str = neg_value if neg_value is not None else ''

    message_elem: ET.Element | None = xmlElement.find(".//Data/Row[@row='1']/Column[@column='3']/Cell")
    message_value: str | None = message_elem.get('value') if message_elem is not None else None
    message: str = message_value if message_value is not None else ''

    return progress, prompt, neg_prompt, message
