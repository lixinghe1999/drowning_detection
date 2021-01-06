import AbstractLinkNamespace 1.0
import DeviceManager 1.0
import PingEnumNamespace 1.0
import QtQuick 2.15
import QtQuick.Controls 2.2
import QtQuick.Controls.Material 2.2
import QtQuick.Layouts 1.3
import SettingsManager 1.0
import Util 1.0

Item {
    id: root

    property var ping: null
    property var serialPortList: null

    signal closeRequest()

    Connections {
        target: ping
        onLinkChanged: {
            // Connection update
            // Update interface to make sure
            switch (ping.link.configuration.type()) {
            case AbstractLinkNamespace.Serial:
                conntype.currentIndex = 0;
                udpLayout.enabled = false;
                serialLayout.enabled = true;
                serialPortsCB.currentText = ping.link.configuration.argsAsConst()[0];
                baudrateBox.currentText = ping.link.configuration.argsAsConst()[1];
                break;
            case AbstractLinkNamespace.Udp:
                conntype.currentIndex = 1;
                udpLayout.enabled = true;
                serialLayout.enabled = false;
                udpIp.text = ping.link.configuration.argsAsConst()[0];
                udpPort.text = ping.link.configuration.argsAsConst()[1];
                break;
            case AbstractLinkNamespace.Ping1DSimulation:
            case AbstractLinkNamespace.Ping360Simulation:
                conntype.currentIndex = 2;
                udpLayout.enabled = false;
                serialLayout.enabled = false;
                break;
            default:
                print("Not valid link.");
                print(ping.link.configuration.name());
                print(ping.link.configuration.type());
                print(ping.link.configuration.argsAsConst());
                return ;
            }
        }
    }

    PingGroupBox {
        id: connectionLayout

        title: "Connection configuration"
        width: parent.width

        GridLayout {
            id: settingsLayout

            anchors.fill: parent
            columns: 5
            rowSpacing: 5
            columnSpacing: 5

            Text {
                text: "Device:"
                color: Material.primary
            }

            ComboBox {
                id: deviceCB

                Layout.fillWidth: true
                textRole: "name"

                //TODO: This model should be provided by someone else and not to be hardcoded here
                model: ListModel {
                    ListElement {
                        name: "Ping1D"
                        deviceId: PingEnumNamespace.PingDeviceType.PING1D
                    }

                    ListElement {
                        name: "Ping360"
                        deviceId: PingEnumNamespace.PingDeviceType.PING360
                    }

                }

            }

            Text {
                text: "Communication:"
                color: Material.primary
            }

            PingComboBox {
                id: conntype

                Layout.columnSpan: 2
                Layout.fillWidth: true
                // Check AbstractLinkNamespace::LinkType for correct index type
                // None = 0, File, Serial, Udp, Tcp..
                // Simulation is done via normal device manager since it does not need user configuration
                model: ["Serial (default)", "UDP", "Simulation"]
                onActivated: {
                    switch (index) {
                    case 0:
                        // Serial
                        udpLayout.enabled = false;
                        serialLayout.enabled = true;
                        break;
                    case 1:
                        // UDP
                        udpLayout.enabled = true;
                        serialLayout.enabled = false;
                        break;
                    case 2:
                        // Simulation
                        udpLayout.enabled = false;
                        serialLayout.enabled = false;
                    }
                }
            }

            RowLayout {
                id: serialLayout

                visible: serialLayout.enabled
                spacing: 5
                Layout.columnSpan: 5

                Text {
                    id: font

                    text: "Serial Port / Baud:"
                    color: Material.primary
                }

                ComboBox {
                    id: serialPortsCB

                    Layout.columnSpan: 2
                    Layout.fillWidth: true
                    model: serialPortList
                    onModelChanged: {
                        var maxWidth = width;
                        for (var i in model) {
                            var modelWidth = (model[i].length + 1) * font.font.pixelSize;
                            maxWidth = maxWidth < modelWidth ? modelWidth : maxWidth;
                        }
                        popup.width = maxWidth;
                    }
                    onPressedChanged: {
                        if (pressed)
                            serialPortList = Util.serialPortList();

                    }
                    Component.onCompleted: serialPortList = Util.serialPortList()
                }

                ComboBox {
                    id: baudrateBox

                    visible: deviceCB.model.get(deviceCB.currentIndex).deviceId == PingEnumNamespace.PingDeviceType.PING1D
                    model: [115200, 9600]
                }

            }

            RowLayout {
                id: udpLayout

                visible: udpLayout.enabled
                spacing: 5
                Layout.columnSpan: 5
                enabled: false

                Text {
                    text: "UDP Host/Port:"
                    color: udpIp.isValid ? Material.primary : Material.color(Material.Error)
                }

                PingTextField {
                    // Check for valid IPV4 (0.0.0.0, 192.168.0.2)
                    // Check for valid host names (potato.com, smashed.potato.com.br, great-potato-of-doom.com)
                    // Check for a valid host name (localhost, raspberrypi, potato)

                    id: udpIp

                    property bool isValid: text.match(/^(?!\.)((^|\.)([1-9]?\d|1\d\d|2(5[0-5]|[0-4]\d))){4}$/gm) || text.match(/^(([a-zA-Z0-9-]+\.){0,5}[a-zA-Z0-9-][a-zA-Z0-9-]+\.[a-zA-Z]{2,63}?)$/gm) || text.match(/^[a-zA-Z0-9-]*$/gm)

                    text: "192.168.2.2"
                    Layout.columnSpan: 2
                    Layout.fillWidth: true
                    onEditingFinished: {
                        if (isValid)
                            connect(AbstractLinkNamespace.Udp, text, udpPort.text);

                    }
                }

                PingTextField {
                    id: udpPort

                    text: "1234"
                    Layout.columnSpan: 2
                    Layout.fillWidth: true

                    validator: IntValidator {
                        bottom: 1
                        top: 65535
                    }

                }

            }

            Button {
                text: "Connect"
                Layout.fillWidth: true
                Layout.columnSpan: 5
                // Disable if using an invalid UDP address
                enabled: conntype.currentIndex === 0 || udpIp.isValid
                onClicked: {
                    var connectionConf = null;
                    var connectionDevice = deviceCB.model.get(deviceCB.currentIndex).deviceId;
                    var connectionType = null;
                    switch (conntype.currentIndex) {
                    case 0:
                        // Serial
                        connectionType = AbstractLinkNamespace.Serial;
                        connectionConf = [serialPortsCB.currentText, baudrateBox.currentText];
                        break;
                    case 1:
                        // UDP
                        connectionType = AbstractLinkNamespace.Udp;
                        connectionConf = [udpIp.text, udpPort.text];
                        break;
                    case 2:
                        // Simulation
                        if (connectionDevice == PingEnumNamespace.PingDeviceType.PING1D)
                            connectionType = AbstractLinkNamespace.Ping1DSimulation;
                        else
                            connectionType = AbstractLinkNamespace.Ping360Simulation;
                        connectionConf = [" ", " "];
                        break;
                    }
                    DeviceManager.connectLinkDirectly(connectionType, connectionConf, connectionDevice);
                    closeRequest();
                }
            }

        }

    }

}
