use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, GenericParam, Generics, parse_macro_input};

/// Derive macro for automatically implementing MPI Equivalence trait that
/// works with generic types.
///
/// # Example
/// ```ignore
/// #[derive(GEquivalence)]
/// struct Point3D {
///     x: f64,
///     y: f64,
///     z: f64,
/// }
///
/// #[derive(GEquivalence)]
/// struct GenericPair<T1, T2> {
///     first: T1,
///     second: T2,
/// }
/// ```
#[proc_macro_derive(GEquivalence)]
pub fn derive_mpi_datatype(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = &input.ident;
    let generics = &input.generics;

    // Extract field information
    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            Fields::Unnamed(_) => {
                return syn::Error::new_spanned(
                    name,
                    "GEquivalence only supports structs with named fields",
                )
                .to_compile_error()
                .into();
            }
            Fields::Unit => {
                return syn::Error::new_spanned(
                    name,
                    "GEquivalence does not support unit structs",
                )
                .to_compile_error()
                .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(
                name,
                "GEquivalence only supports structs",
            )
            .to_compile_error()
            .into();
        }
    };

    if fields.is_empty() {
        return syn::Error::new_spanned(
            name,
            "GEquivalence requires at least one field",
        )
        .to_compile_error()
        .into();
    }

    // Generate field names for offset_of! and type inference
    let field_names: Vec<_> = fields.iter().map(|f| &f.ident).collect();
    let field_count = fields.len();

    // Build the impl generics with Equivalence bounds
    let generics_with_bounds = add_trait_bounds(generics);
    let (impl_generics, ty_generics, where_clause) =
        generics_with_bounds.split_for_impl();

    // Create count tokens (one per field)
    let counts = vec![quote! { 1 as ::mpi::Count }; field_count];

    // Implementation 
    // This following is based on the comment in Rust lang Forums:
    // https://users.rust-lang.org/t/any-way-to-create-a-generic-static/73556/2
    // code :
    // https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=b7630bd5c87ae0147e099ea2bf7010e9
    // Initial version changed to
    //   - remove typemap,
    //   - use standard library instead of parking_lot
    //   - use of 'static
    let expanded = quote! {
        unsafe impl #impl_generics ::mpi::traits::Equivalence for #name #ty_generics #where_clause {
            type Out = ::mpi::datatype::DatatypeRef<'static>;

            fn equivalent_datatype() -> Self::Out {
                use ::std::sync::Mutex;
                use ::std::collections::HashMap;
                use ::std::any::TypeId;
                use ::std::sync::LazyLock;
                use ::mpi::datatype::{UncommittedDatatypeRef, UserDatatype};

                static DTYPE_MAP: LazyLock<Mutex<HashMap<TypeId, &'static UserDatatype>>> =
                    LazyLock::new(|| Mutex::new(HashMap::new()));

                let mut map = DTYPE_MAP.lock().unwrap();
                let type_id = TypeId::of::<Self>();

                let datatype = map
                    .entry(type_id)
                    .or_insert_with(|| {
                        // Create arrays for structured datatype
                        let counts = [#(#counts),*];

                        let displacements = [#(
                            ::std::mem::offset_of!(Self, #field_names) as ::mpi::Address
                        ),*];

                        let datatypes = [#(
                            UncommittedDatatypeRef::from(
                                Self::__get_field_type_helper(|s: &Self| &s.#field_names)
                            )
                        ),*];

                        let datatype = UserDatatype::structured(
                            &counts,
                            &displacements,
                            &datatypes,
                        );

                        Box::leak(Box::new(datatype))
                    });

                datatype.as_ref()
            }
        }

        impl #impl_generics #name #ty_generics #where_clause {
            #[doc(hidden)]
            fn __get_field_type_helper<T: ::mpi::traits::Equivalence>(
                _f: impl FnOnce(&Self) -> &T
            ) -> T::Out {
                T::equivalent_datatype()
            }
        }
    };

    TokenStream::from(expanded)
}

/// Add `Equivalence + Clone + 'static` bounds to all type parameters
fn add_trait_bounds(generics: &Generics) -> Generics {
    let mut generics = generics.clone();

    for param in &mut generics.params {
        if let GenericParam::Type(type_param) = param {
            type_param.bounds.push(syn::parse_quote!(
                ::mpi::traits::Equivalence<
                    Out = ::mpi::datatype::DatatypeRef<'static>,
                >
            ));
            type_param.bounds.push(syn::parse_quote!(Clone));
            type_param.bounds.push(syn::parse_quote!('static));
        }
    }

    generics
}
