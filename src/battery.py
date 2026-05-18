from __future__ import annotations
from dataclasses import dataclass, field

@dataclass

class Battery:
    """Simple battery model for dispatch simulations.

    The battery stores energy internally in kWh. State of charge (SOC) is
    derived from stored energy and exposed as a property.
    """

    capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    min_soc: float = 0.0
    max_soc: float = 1.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    initial_soc: float = 0.5
    energy_kwh: float = field(init=False)

    def __post_init__(self) -> None:
        self._validate_inputs()
        self.energy_kwh = self._clip_energy(self.initial_soc * self.capacity_kwh)

    @property
    def soc(self) -> float:
        """Current state of charge in [0, 1]."""
        return self.energy_kwh / self.capacity_kwh

    @property
    def min_energy_kwh(self) -> float:
        return self.min_soc * self.capacity_kwh

    @property
    def max_energy_kwh(self) -> float:
        return self.max_soc * self.capacity_kwh

    def available_charge_kw(self, dt_hours: float) -> float:
        """Maximum feasible charging power for this timestep."""
        self._validate_timestep(dt_hours)
        room_kwh = self.max_energy_kwh - self.energy_kwh
        if room_kwh <= 0.0:
            return 0.0

        # Grid-side charging power that fits after efficiency losses.
        energy_in_limit_kw = room_kwh / (self.charge_efficiency * dt_hours)
        return max(0.0, min(self.max_charge_kw, energy_in_limit_kw))

    def available_discharge_kw(self, dt_hours: float) -> float:
        """Maximum feasible discharging power for this timestep."""
        self._validate_timestep(dt_hours)
        usable_kwh = self.energy_kwh - self.min_energy_kwh
        if usable_kwh <= 0.0:
            return 0.0

        # Grid-side discharge power that can be sustained given internal energy.
        energy_out_limit_kw = usable_kwh * self.discharge_efficiency / dt_hours
        return max(0.0, min(self.max_discharge_kw, energy_out_limit_kw))

    def apply_power(self, requested_kw: float, dt_hours: float) -> float:
        """Apply a requested battery power and update internal energy.

        Sign convention:
            requested_kw < 0: charge request
            requested_kw > 0: discharge request
            requested_kw = 0: idle

        Returns the actual applied power at the battery terminals after clipping
        to physical constraints.
        """
        self._validate_timestep(dt_hours)

        if requested_kw < 0.0:
            actual_charge_kw = min(-requested_kw, self.available_charge_kw(dt_hours))
            self.energy_kwh += actual_charge_kw * dt_hours * self.charge_efficiency
            self.energy_kwh = self._clip_energy(self.energy_kwh)
            return -actual_charge_kw

        if requested_kw > 0.0:
            actual_discharge_kw = min(requested_kw, self.available_discharge_kw(dt_hours))
            self.energy_kwh -= (actual_discharge_kw * dt_hours) / self.discharge_efficiency
            self.energy_kwh = self._clip_energy(self.energy_kwh)
            return actual_discharge_kw

        return 0.0

    def _clip_energy(self, energy_kwh: float) -> float:
        return min(max(energy_kwh, self.min_energy_kwh), self.max_energy_kwh)

    def _validate_inputs(self) -> None:
        if self.capacity_kwh <= 0.0:
            raise ValueError("capacity_kwh must be positive")
        if self.max_charge_kw <= 0.0:
            raise ValueError("max_charge_kw must be positive")
        if self.max_discharge_kw <= 0.0:
            raise ValueError("max_discharge_kw must be positive")
        if not 0.0 <= self.min_soc <= 1.0:
            raise ValueError("min_soc must be in [0, 1]")
        if not 0.0 <= self.max_soc <= 1.0:
            raise ValueError("max_soc must be in [0, 1]")
        if self.min_soc >= self.max_soc:
            raise ValueError("min_soc must be smaller than max_soc")
        if not 0.0 < self.charge_efficiency <= 1.0:
            raise ValueError("charge_efficiency must be in (0, 1]")
        if not 0.0 < self.discharge_efficiency <= 1.0:
            raise ValueError("discharge_efficiency must be in (0, 1]")
        if not self.min_soc <= self.initial_soc <= self.max_soc:
            raise ValueError("initial_soc must lie between min_soc and max_soc")

    @staticmethod
    def _validate_timestep(dt_hours: float) -> None:
        if dt_hours <= 0.0:
            raise ValueError("dt_hours must be positive")