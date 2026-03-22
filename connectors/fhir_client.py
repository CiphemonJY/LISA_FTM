"""
FHIR R4 Connector for LISA_FTM
==============================
Unified FHIR R4 client for Epic, Cerner (Oracle Health), MEDITECH,
Athenahealth, and any FHIR R4-compliant EHR.

Usage:
    from connectors.fhir_client import FHIRClient

    client = FHIRClient(
        base_url="https://epic.hospital.org/fhir/r4",
        api_token="...",       # Bearer token or OAuth2 access token
    )

    # Get oncology patients
    for patient in client.get_oncology_patients():
        print(patient.as_text())
"""
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator, Optional

import requests

log = logging.getLogger("fhir-connector")

# Standard FHIR R4 resource type → our wrapper class
FHIR_PATIENT_RESOURCE = "Patient"
FHIR_CONDITION = "Condition"
FHIR_MEDICATION_REQUEST = "MedicationRequest"
FHIR_OBSERVATION_LAB = "Observation"
FHIR_OBSERVATION_VITAL = "Observation"
FHIR_DIAGNOSTIC_REPORT = "DiagnosticReport"
FHIR_DOCUMENT_REFERENCE = "DocumentReference"

# Common oncology ICD-10 codes (primary sites, C00-C97)
ONCOLOGY_ICD10_PREFIXES = ("C",)

# Common LOINC codes for relevant labs
LOINC_CEA = "85319"
LOINC_CA125 = "1005-1"
LOINC_CA19_9 = "83358-5"
LOINC_LDH = "2532-0"
LOINC_ALBUMIN = "1751-7"
LOINC_HGB = "718-7"
LOINC_WBC = "6690-2"
LOINC_PLT = "777-3"


@dataclass
class FHIRPatient:
    """Wrapper for FHIR Patient resource with plain-text conversion."""

    resource_id: str
    name: str
    age: Optional[int] = None
    sex: str = "Unknown"
    race: Optional[str] = None
    ethnicity: Optional[str] = None

    # FHIR resource references
    conditions: list[dict] = field(default_factory=list)
    medications: list[dict] = field(default_factory=list)
    lab_results: list[dict] = field(default_factory=list)
    vital_signs: list[dict] = field(default_factory=list)
    genomic_reports: list[dict] = field(default_factory=list)
    clinical_notes: list[str] = field(default_factory=list)

    def add_condition(self, condition: dict):
        self.conditions.append(condition)

    def add_medication(self, med: dict):
        self.medications.append(med)

    def add_lab_result(self, lab: dict):
        self.lab_results.append(lab)

    def add_vital_sign(self, vital: dict):
        self.vital_signs.append(vital)

    def add_genomic_report(self, report: dict):
        self.genomic_reports.append(report)

    def add_clinical_note(self, note: str):
        self.clinical_notes.append(note)

    def as_text(self) -> str:
        """
        Convert the full patient record to plain text for LISA_FTM training.
        This is the core method — output format determines what the model learns.
        """
        lines = [f"Patient: {self.name}"]

        if self.age:
            lines.append(f"Age: {self.age}-year-old")
        if self.sex:
            lines.append(f"Sex: {self.sex}")
        if self.race:
            lines.append(f"Race: {self.race}")
        if self.ethnicity:
            lines.append(f"Ethnicity: {self.ethnicity}")

        # Conditions / Diagnoses
        if self.conditions:
            lines.append("Diagnoses:")
            for c in self.conditions[:10]:  # limit to top 10
                code = c.get("code", "")
                name = c.get("name", "")
                status = c.get("clinicalStatus", "")
                lines.append(f"  - {name} ({code}) [{status}]")

        # Medications
        if self.medications:
            lines.append("Medications:")
            for m in self.medications[:10]:
                drug = m.get("medication", "")
                status = m.get("status", "")
                lines.append(f"  - {drug} ({status})")

        # Lab results
        if self.lab_results:
            lines.append("Labs:")
            for lab in self.lab_results[:15]:
                test = lab.get("test", "")
                value = lab.get("value", "")
                unit = lab.get("unit", "")
                date = lab.get("date", "")
                lines.append(f"  - {test}: {value} {unit} ({date})")

        # Vital signs
        if self.vital_signs:
            lines.append("Vitals:")
            for v in self.vital_signs[-5:]:  # most recent 5
                vital_type = v.get("type", "")
                value = v.get("value", "")
                unit = v.get("unit", "")
                date = v.get("date", "")
                lines.append(f"  - {vital_type}: {value} {unit} ({date})")

        # Genomic reports
        if self.genomic_reports:
            lines.append("Genomic markers:")
            for g in self.genomic_reports:
                marker = g.get("marker", "")
                result = g.get("result", "")
                assay = g.get("assay", "")
                lines.append(f"  - {marker}: {result} ({assay})")

        # Clinical notes (truncated)
        if self.clinical_notes:
            lines.append("Clinical notes:")
            for note in self.clinical_notes[:3]:
                snippet = note[:500] if len(note) > 500 else note
                lines.append(f"  {snippet}")

        return "\n".join(lines)

    def __repr__(self):
        return f"FHIRPatient(id={self.resource_id}, name={self.name})"


class FHIRCodeMapper:
    """
    Maps FHIR codes (LOINC, ICD-10, RxNorm, SNOMED) to human-readable strings.
    In production you'd use the FHIR Terminology Service.
    """

    @staticmethod
    def icd10_lookup(code: str) -> str:
        """Return a human-readable ICD-10 description."""
        # In production: call FHIR CodeSystem/$lookup
        return code

    @staticmethod
    def loinc_lookup(code: str) -> str:
        """Return a human-readable LOINC test name."""
        LOINC_NAMES = {
            LOINC_CEA: "CEA",
            LOINC_CA125: "CA-125",
            LOINC_CA19_9: "CA 19-9",
            LOINC_LDH: "LDH",
            LOINC_ALBUMIN: "Albumin",
            LOINC_HGB: "Hemoglobin",
            LOINC_WBC: "WBC",
            LOINC_PLT: "Platelets",
        }
        return LOINC_NAMES.get(code, code)

    @staticmethod
    def rxnorm_lookup(code: str) -> str:
        """Return a human-readable RxNorm medication name."""
        # In production: call RxNav API
        return code


class FHIRClient:
    """
    Unified FHIR R4 client for any FHIR-compliant EHR.

    Handles:
    - Bearer token authentication
    - OAuth2 / Smart on FHIR authentication
    - Basic auth (legacy systems)
    - Pagination (FHIR Bundle responses)
    - Rate limiting (429 backoff)
    - Error handling
    """

    def __init__(
        self,
        base_url: str,
        api_token: str = "",
        auth_type: str = "bearer",  # bearer | basic | smart
        client_id: str = "",
        client_secret: str = "",
        scope: str = "system/*.read",
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.mapper = FHIRCodeMapper()
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json",
        })

        if auth_type == "bearer":
            self._session.headers["Authorization"] = f"Bearer {api_token}"
        elif auth_type == "basic":
            import base64
            creds = base64.b64encode(f"{api_token}:".encode()).decode()
            self._session.headers["Authorization"] = f"Basic {creds}"
        elif auth_type == "smart":
            self._do_smart_auth(client_id, client_secret, scope)

        log.info(f"FHIRClient initialized: {base_url}")

    def _do_smart_auth(self, client_id: str, client_secret: str, scope: str):
        """OAuth2 Smart on FHIR launch flow."""
        # In production: implement full OAuth2 PKCE flow
        # This is the token endpoint call
        token_url = f"{self.base_url}/oauth2/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": scope,
        }
        resp = self._session.post(token_url, data=data, timeout=self.timeout)
        resp.raise_for_status()
        token = resp.json().get("access_token")
        self._session.headers["Authorization"] = f"Bearer {token}"

    def _get(self, path: str, params: dict = None) -> dict:
        """Execute a FHIR GET with rate limiting and pagination."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        all_entries = []

        while url:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 30))
                log.warning(f"Rate limited, waiting {retry_after}s")
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            bundle = resp.json()

            entries = bundle.get("entry", [])
            all_entries.extend(entries)

            # FHIR pagination — link URL to next page
            links = {l["relation"]: l["url"] for l in bundle.get("link", [])}
            url = links.get("next", None)
            params = None  # next link already has params

            # Safety limit
            if len(all_entries) > 10000:
                log.warning(f"Reached pagination limit at {len(all_entries)} entries")
                break

        return {"entry": all_entries}

    # ─── Patient queries ───────────────────────────────────────────────────

    def get_patients(
        self,
        name: str = "",
        identifier: str = "",
        _count: int = 100,
    ) -> Iterator[FHIRPatient]:
        """
        Fetch patients, optionally filtered by name or identifier.
        """
        params = {"_count": _count}
        if name:
            params["name"] = name
        if identifier:
            params["identifier"] = identifier

        bundle = self._get("Patient", params)
        for entry in bundle.get("entry", []):
            resource = entry["resource"]
            yield self._parse_patient(resource)

    def get_oncology_patients(
        self,
        icd10_prefix: str = "C",
        _count: int = 500,
    ) -> Iterator[FHIRPatient]:
        """
        Get all patients with oncology-related diagnoses.
        Uses ICD-10 codes starting with C00-C97.
        """
        # Search for conditions with oncology ICD-10 codes
        bundle = self._get(
            "Condition",
            params={
                "_count": _count,
                "clinical-status": "active,recurrence,relapse",
                "code": f"http://hl7.org/fhir/sid/icd-10-cm|{icd10_prefix}",
            },
        )

        # Group by patient
        patient_map: dict[str, FHIRPatient] = {}
        for entry in bundle.get("entry", []):
            resource = entry["resource"]
            patient_ref = resource.get("subject", {}).get("reference", "")
            if not patient_ref:
                continue

            patient_id = patient_ref.replace("Patient/", "")
            if patient_id not in patient_map:
                patient_map[patient_id] = FHIRPatient(
                    resource_id=patient_id,
                    name=f"Patient_{patient_id}",
                )

            code = resource.get("code", {})
            code_coding = code.get("coding", [{}])[0]
            patient_map[patient_id].add_condition({
                "code": code_coding.get("code", ""),
                "name": code_coding.get("display", ""),
                "clinicalStatus": resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", ""),
            })

        yield from patient_map.values()

    # ─── Per-patient queries ───────────────────────────────────────────────

    def get_demographics(self, patient_id: str) -> FHIRPatient:
        """Fetch basic demographics for one patient."""
        bundle = self._get(f"Patient/{patient_id}")
        entries = bundle.get("entry", [])
        if not entries:
            return FHIRPatient(resource_id=patient_id, name=f"Patient_{patient_id}")
        return self._parse_patient(entries[0]["resource"])

    def get_conditions(self, patient_id: str) -> list[dict]:
        """Fetch all conditions (diagnoses) for a patient."""
        bundle = self._get(
            "Condition",
            params={"patient": patient_id, "_count": 200},
        )
        results = []
        for entry in bundle.get("entry", []):
            resource = entry["resource"]
            code = resource.get("code", {})
            code_coding = code.get("coding", [{}])[0]
            results.append({
                "code": code_coding.get("code", ""),
                "name": code_coding.get("display", ""),
                "clinicalStatus": resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", ""),
                "verificationStatus": resource.get("verificationStatus", {}).get("coding", [{}])[0].get("code", ""),
            })
        return results

    def get_medications(self, patient_id: str) -> list[dict]:
        """Fetch medication history for a patient."""
        bundle = self._get(
            "MedicationRequest",
            params={"patient": patient_id, "_count": 200},
        )
        results = []
        for entry in bundle.get("entry", []):
            resource = entry["resource"]
            med = resource.get("medicationCodeableConcept", {})
            coding = med.get("coding", [{}])[0]
            results.append({
                "medication": coding.get("display", ""),
                "code": coding.get("code", ""),
                "status": resource.get("status", ""),
                "intent": resource.get("intent", ""),
            })
        return results

    def get_lab_results(self, patient_id: str, category: str = "laboratory") -> list[dict]:
        """Fetch lab results for a patient."""
        bundle = self._get(
            "Observation",
            params={
                "patient": patient_id,
                "category": category,
                "_count": 500,
                "_sort": "-date",
            },
        )
        results = []
        for entry in bundle.get("entry", []):
            resource = entry["resource"]
            code = resource.get("code", {})
            coding = code.get("coding", [{}])[0]
            value = resource.get("valueQuantity", {})
            results.append({
                "test": self.mapper.loinc_lookup(coding.get("code", "")),
                "code": coding.get("code", ""),
                "value": value.get("value", ""),
                "unit": value.get("unit", ""),
                "date": resource.get("effectiveDateTime", ""),
            })
        return results

    def get_vital_signs(self, patient_id: str) -> list[dict]:
        """Fetch vital signs for a patient."""
        return self.get_lab_results(patient_id, category="vital-signs")

    def get_genomic_reports(self, patient_id: str) -> list[dict]:
        """Fetch genomic/lab reports for a patient."""
        bundle = self._get(
            "DiagnosticReport",
            params={
                "patient": patient_id,
                "category": "GE",
                "_count": 100,
            },
        )
        results = []
        for entry in bundle.get("entry", []):
            resource = entry["resource"]
            results.append({
                "report": resource.get("code", {}).get("text", ""),
                "status": resource.get("status", ""),
                "date": resource.get("effectiveDateTime", ""),
            })
        return results

    def get_clinical_notes(self, patient_id: str) -> list[str]:
        """Fetch clinical notes (DocumentReference) for a patient."""
        bundle = self._get(
            "DocumentReference",
            params={
                "patient": patient_id,
                "status": "current",
                "_count": 50,
            },
        )
        results = []
        for entry in bundle.get("entry", []):
            resource = entry["resource"]
            content = resource.get("content", [{}])[0]
            attachment = content.get("attachment", {})
            text = attachment.get("data", "")
            if text:
                import base64 as b64
                decoded = b64.b64decode(text).decode("utf-8", errors="replace")
                results.append(decoded)
        return results

    # ─── Internal parsers ──────────────────────────────────────────────────

    def _parse_patient(self, resource: dict) -> FHIRPatient:
        """Parse a FHIR Patient resource into FHIRPatient."""
        patient_id = resource.get("id", "")

        # Name
        name_parts = resource.get("name", [{}])[0]
        given = " ".join(name_parts.get("given", []))
        family = name_parts.get("family", "")
        full_name = f"{given} {family}".strip() or f"Patient_{patient_id}"

        # Demographics
        gender = resource.get("gender", "unknown")
        gender_map = {"male": "M", "female": "F", "other": "Other", "unknown": "Unknown"}
        sex = gender_map.get(gender, "Unknown")

        # Birthdate → age
        birth_date = resource.get("birthDate", "")
        age = None
        if birth_date:
            try:
                birth = datetime.strptime(birth_date, "%Y-%m-%d")
                age = (datetime.now() - birth).days // 365
            except (ValueError, TypeError):
                pass

        # Race / ethnicity
        ext = resource.get("extension", [])
        race = None
        ethnicity = None
        for e in ext:
            if "race" in e.get("url", "").lower():
                race = e.get("valueString", "")
            if "ethnicity" in e.get("url", "").lower():
                ethnicity = e.get("valueString", "")

        return FHIRPatient(
            resource_id=patient_id,
            name=full_name,
            age=age,
            sex=sex,
            race=race,
            ethnicity=ethnicity,
        )

    def get_full_patient_record(self, patient_id: str) -> FHIRPatient:
        """
        Convenience: fetch all data for one patient and return as FHIRPatient.
        Calls get_demographics, get_conditions, get_medications, get_labs,
        get_vitals, get_genomic_reports, get_clinical_notes.
        """
        patient = self.get_demographics(patient_id)

        for cond in self.get_conditions(patient_id):
            patient.add_condition(cond)

        for med in self.get_medications(patient_id):
            patient.add_medication(med)

        for lab in self.get_lab_results(patient_id):
            patient.add_lab_result(lab)

        for vital in self.get_vital_signs(patient_id):
            patient.add_vital_sign(vital)

        for report in self.get_genomic_reports(patient_id):
            patient.add_genomic_report(report)

        for note in self.get_clinical_notes(patient_id):
            patient.add_clinical_note(note)

        return patient


def demo():
    """
    Demo: fetch oncology patients from a FHIR server.
    Usage:
        python -m connectors.fhir_client --url https://epic.hospital.org/fhir/r4 --token TOKEN
    """
    import argparse

    parser = argparse.ArgumentParser(description="FHIR R4 Connector Demo")
    parser.add_argument("--url", required=True, help="FHIR base URL")
    parser.add_argument("--token", default="", help="API token / Bearer token")
    parser.add_argument("--patient-id", help="Specific patient ID to fetch")
    args = parser.parse_args()

    client = FHIRClient(args.url, api_token=args.token)

    if args.patient_id:
        patient = client.get_full_patient_record(args.patient_id)
        print(patient.as_text())
    else:
        count = 0
        for patient in client.get_oncology_patients():
            print(f"--- Patient {count + 1} ---")
            print(patient.as_text())
            print()
            count += 1
            if count >= 5:
                break


if __name__ == "__main__":
    demo()
