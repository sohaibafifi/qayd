//! A tiny read-only XML DOM built on quick-xml.
//!
//! XCSP3 mixes attributes, text, and children freely, which is awkward for
//! `serde`. Parsing into a small node tree and walking it is simpler and keeps
//! the front-end honest about exactly which elements it interprets.

use quick_xml::events::{BytesStart, Event};
use quick_xml::reader::Reader;

/// One XML element: its tag, attributes, concatenated text, and children.
#[derive(Debug, Default, Clone)]
pub struct Node {
    /// Element tag name.
    pub tag: String,
    /// `(name, value)` attribute pairs.
    pub attrs: Vec<(String, String)>,
    /// Concatenated text content (direct children's text is not folded in).
    pub text: String,
    /// Child elements, in document order.
    pub children: Vec<Node>,
}

impl Node {
    /// Value of attribute `name`, if present.
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    /// First child element with the given tag.
    pub fn child(&self, tag: &str) -> Option<&Node> {
        self.children.iter().find(|c| c.tag == tag)
    }

    /// All child elements with the given tag.
    pub fn children_named<'a>(&'a self, tag: &'a str) -> impl Iterator<Item = &'a Node> {
        self.children.iter().filter(move |c| c.tag == tag)
    }

    /// Text content, trimmed.
    pub fn trimmed(&self) -> &str {
        self.text.trim()
    }
}

fn node_from(e: &BytesStart) -> Result<Node, String> {
    let tag = std::str::from_utf8(e.name().as_ref())
        .map_err(|e| e.to_string())?
        .to_string();
    let mut attrs = Vec::new();
    for a in e.attributes() {
        let a = a.map_err(|e| e.to_string())?;
        let key = std::str::from_utf8(a.key.as_ref())
            .map_err(|e| e.to_string())?
            .to_string();
        let value = std::str::from_utf8(&a.value)
            .map_err(|e| e.to_string())?
            .to_string();
        attrs.push((key, value));
    }
    Ok(Node {
        tag,
        attrs,
        text: String::new(),
        children: Vec::new(),
    })
}

/// Parse XML into the root [`Node`].
pub fn parse(xml: &str) -> Result<Node, String> {
    let mut reader = Reader::from_str(xml);
    let mut stack: Vec<Node> = Vec::new();
    let mut root: Option<Node> = None;

    let attach = |node: Node, stack: &mut Vec<Node>, root: &mut Option<Node>| {
        if let Some(top) = stack.last_mut() {
            top.children.push(node);
        } else {
            *root = Some(node);
        }
    };

    loop {
        match reader.read_event().map_err(|e| e.to_string())? {
            Event::Start(e) => stack.push(node_from(&e)?),
            Event::Empty(e) => {
                let node = node_from(&e)?;
                attach(node, &mut stack, &mut root);
            }
            Event::Text(e) => {
                let t = std::str::from_utf8(&e).map_err(|e| e.to_string())?;
                if let Some(top) = stack.last_mut() {
                    top.text.push_str(t);
                }
            }
            Event::End(_) => {
                let node = stack.pop().ok_or("unbalanced end tag")?;
                attach(node, &mut stack, &mut root);
            }
            Event::Eof => break,
            _ => {}
        }
    }
    root.ok_or_else(|| "no root element".to_string())
}
