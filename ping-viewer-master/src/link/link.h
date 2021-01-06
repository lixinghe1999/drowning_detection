#pragma once

#include <QObject>

#include <memory>

#include "abstractlink.h"

class LinkConfiguration;

/**
 * @brief Top class for all links
 *
 */
class Link : public QObject {
    Q_OBJECT
public:
    /**
     * @brief Construct a new Link object
     *
     * @param linkType
     * @param name
     * @param parent
     */
    Link(LinkType linkType = LinkType::None, QString name = QString(), QObject* parent = nullptr);

    /**
     * @brief Construct a new Link object
     *
     * @param linkConfiguration
     * @param parent
     */
    Link(const LinkConfiguration& linkConfiguration, QObject* parent = nullptr);
    ~Link();

    /**
     * @brief This will handle the link used
     *
     * @return AbstractLink*
     */
    AbstractLink* self() { return _abstractLink.get(); };

private:
    std::unique_ptr<AbstractLink> _abstractLink;
};
