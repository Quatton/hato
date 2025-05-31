from typing import List
from pydantic import BaseModel


class Prefecture(BaseModel):
    name: str
    kana: str
    code: str


class City(BaseModel):
    name: str
    kana: str
    code: str


class NameKana(BaseModel):
    name: str
    kana: str


class Address(BaseModel):
    fullAddress: str
    country: str
    prefecture: Prefecture
    city: City
    oaza: NameKana
    aza: NameKana
    detail1: NameKana | None = None


class PanoAddress(BaseModel):
    panoId: str
    lat: float
    lng: float
    address: Address


class PanoAddressDataset(BaseModel):
    customCoordinates: List[PanoAddress]
