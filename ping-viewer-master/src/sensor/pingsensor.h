#pragma once

#include "sensor.h"

/**
 * @brief Abstract ping sensors
 *
 */
class PingSensor : public Sensor {
    Q_OBJECT
public:
    /**
     * @brief Construct a new Ping Sensor object
     *
     */
    PingSensor(PingDeviceType pingDeviceType);

    /**
     * @brief Destroy the Ping Sensor object
     *
     */
    ~PingSensor();

    /**
     * @brief debug function
     */
    virtual void printStatus() const;

    /**
     * @brief Add new connection
     *
     * @param connType connection type
     * @param connString arguments for the new connection
     */
    Q_INVOKABLE virtual void connectLink(AbstractLinkNamespace::LinkType connType, const QStringList& connString) = 0;

    /**
     * @brief Get device source ID
     *
     * @return uint8_t source id
     */
    uint8_t srcId() const { return _commonVariables.srcId; }
    Q_PROPERTY(int srcId READ srcId NOTIFY srcIdChanged)

    /**
     * @brief Return destination ID
     *
     * @return uint8_t destination ID
     */
    uint8_t dstId() const { return _commonVariables.dstId; }
    Q_PROPERTY(int dstId READ dstId NOTIFY dstIdChanged)

    // TODO: move functions from snake case to camel case
    /**
     * @brief Return firmware major version
     *
     * @return uint8_t firmware major version number
     */
    uint8_t firmware_version_major() const { return _commonVariables.deviceInformation.firmware_version_major; }
    Q_PROPERTY(int firmware_version_major READ firmware_version_major NOTIFY firmwareVersionMajorChanged)

    /**
     * @brief Return firmware minor version
     *
     * @return uint8_t firmware minor version number
     */
    uint16_t firmware_version_minor() const { return _commonVariables.deviceInformation.firmware_version_minor; }
    Q_PROPERTY(int firmware_version_minor READ firmware_version_minor NOTIFY firmwareVersionMinorChanged)

    /**
     * @brief Return firmware patch version
     *
     * @return uint8_t firmware patch version number
     */
    uint16_t firmware_version_patch() const { return _commonVariables.deviceInformation.firmware_version_patch; }
    Q_PROPERTY(int firmware_version_patch READ firmware_version_patch NOTIFY firmwareVersionPatchChanged)

    /**
     * @brief Return device type number
     *
     * @return uint8_t Device type number
     */
    uint8_t device_type() const { return _commonVariables.deviceInformation.device_type; }
    Q_PROPERTY(int device_type READ device_type NOTIFY deviceTypeChanged)

    /**
     * @brief Return device model number
     *
     * @return uint8_t Device model number
     */
    uint8_t device_revision() const { return _commonVariables.deviceInformation.device_revision; }
    Q_PROPERTY(int device_revision READ device_revision NOTIFY deviceRevisionChanged)

    /**
     * @brief Return last ascii_text message
     *
     * @return QString
     */
    QString asciiText() const { return _commonVariables.ascii_text; }
    Q_PROPERTY(QString ascii_text READ asciiText NOTIFY asciiTextChanged)

    /**
     * @brief Return last nack message
     *
     * @return QString
     */
    QString nackMessage() const { return _commonVariables.nack_msg; }
    Q_PROPERTY(QString nack_message READ nackMessage NOTIFY nackMsgChanged)

    /**
     * @brief Return number of parser errors
     *
     * @return int
     */
    int parserErrors() const { return _parser ? _parser->errors : 0; }
    Q_PROPERTY(int parser_errors READ parserErrors NOTIFY parserErrorsChanged)

    /**
     * @brief Return number of successfully parsed messages
     *
     * @return int
     */
    int parsedMsgs() const { return _parser ? _parser->parsed : 0; }
    Q_PROPERTY(int parsed_msgs READ parsedMsgs NOTIFY parsedMsgsChanged)

    /**
     * @brief Return the number of messages that we requested and did not received
     *
     * @return int
     */
    int lostMessages() { return _lostMessages; }
    Q_PROPERTY(int lost_messages READ lostMessages NOTIFY lostMessagesChanged)

    /**
     * @brief Request message id
     *
     * @param id
     */
    Q_INVOKABLE virtual void request(int id) const;

    /**
     * @brief Do firmware sensor update
     *
     * @param fileUrl firmware file path
     * @param sendPingGotoBootloader Use "goto bootloader" message
     * @param baud baud rate value
     * @param verify this variable is true when all
     */
    Q_INVOKABLE virtual void firmwareUpdate(
        QString fileUrl, bool sendPingGotoBootloader = true, int baud = 57600, bool verify = true)
        = 0;

signals:
    void asciiTextChanged();
    void deviceRevisionChanged();
    void deviceTypeChanged();
    void dstIdChanged();
    void firmwareVersionMajorChanged();
    void firmwareVersionMinorChanged();
    void firmwareVersionPatchChanged();
    void lostMessagesChanged();
    void nackMsgChanged();
    void parsedMsgsChanged();
    void parserErrorsChanged();
    void protocolVersionMajorChanged();
    void protocolVersionMinorChanged();
    void protocolVersionPatchChanged();
    void srcIdChanged();

protected:
    /**
     * @brief Handle new ping protocol messages
     *
     * @param msg
     */
    void handleMessagePrivate(const ping_message& msg);

    /**
     * @brief Handle new ping protocol messages
     *
     * @param msg
     */
    virtual void handleMessage(const ping_message& msg) { Q_UNUSED(msg) };

    /**
     * @brief Print specific information about a specific sensor
     *  Information will be printed with pingStatus
     */
    virtual void printSensorInformation() const = 0;

    /**
     * @brief Write a message to link
     *
     * @param msg
     */
    void writeMessage(const ping_message& msg) const;

    // Common variables between all ping devices
    struct CommonVariables {
        QString ascii_text;
        uint8_t dstId {0};

        struct {
            bool initialized = false;
            uint8_t device_revision {0};
            uint8_t device_type {0};
            uint8_t firmware_version_major {0};
            uint8_t firmware_version_minor {0};
            uint8_t firmware_version_patch {0};
        } deviceInformation;

        QString nack_msg;
        uint8_t protocol_version_major {0};
        uint8_t protocol_version_minor {0};
        uint8_t protocol_version_patch {0};
        uint8_t srcId {0};

        /**
         * @brief Reset variables to default values
         *
         */
        inline void reset() { *this = {}; }
    } _commonVariables;

    int _lostMessages {0};

private:
    Q_DISABLE_COPY(PingSensor)
};
