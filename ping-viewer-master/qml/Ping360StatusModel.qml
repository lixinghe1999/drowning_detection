import QtQml.Models 2.12
import QtQuick 2.15

Item {
    property var title: "Ping360 specific data:"
    property var sensor: null
    // It's not possible to use DelegateModel without parent
    property var model

    Timer {
        interval: 200
        running: true
        repeat: true
        // Limit the frequency update of the model from the sensor properties
        onTriggered: {
            if (!sensor)
                return ;

            delegateModel.model = ["Range (m): " + sensor.range.toFixed(2), "Sample period (ticks): " + sensor.sample_period, "Sample period (ns): " + sensor.sample_period * 25, "Number of samples (#): " + sensor.number_of_points, "Profile frequency (Hz): " + sensor.profileFrequency.toFixed(2), "Ping (#): " + sensor.ping_number, "Angle (grad): " + sensor.angle, "Angle Offset (grad): " + sensor.angle_offset, "Transmit frequency (kHz): " + sensor.transmit_frequency, "Transmit duration (μs): " + sensor.transmit_duration, "Transmit duration maximum (μs): " + sensor.transmitDurationMax, "Gain (setting): " + sensor.gain_setting, "Speed of sound (m/s): " + sensor.speed_of_sound];
        }
    }

    model: DelegateModel {
        id: delegateModel

        delegate: Text {
            text: modelData
            color: "white"
            font.pointSize: 8
        }

    }

}
